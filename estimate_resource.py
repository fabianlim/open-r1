from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
from trl import ScriptArguments, TrlParser
from trl.data_utils import apply_chat_template
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from accelerate import Accelerator
from trl.trainer.utils import selective_log_softmax

import datetime

# verify we have FSDP activation support ready by importing:
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

import os

from open_r1.grpo import SYSTEM_PROMPT

from vllm import LLM, SamplingParams

from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import math
import time, json
from tqdm import trange
import numpy as np

DATASET_FILE = 'dataset-{}-{}.json'
METRICS_VLLM_FILE = 'metrics_vllm-{}.json'
METRICS_TRAIN_FILE = 'metrics_train-{}.json'

@dataclass
class Arguments(ScriptArguments):
    """
    Script arguments
    """
    model_name: str = "Qwen/Qwen2.5-Math-7B"
    dataset_name: str = "GAIR/LIMO"
    results_dir: str = 'resource_estimation/trial'
    run_vllm: bool = False
    run_training: bool = False
    num_devices: int = 1
    num_generations: int = 1
    train_batch_size: int = 1
    vllm_tensor_parallel_size: int = 1
    use_flash_attn: bool = False

def estimate_vllm(
    model: str,
    dataset: Dataset,
    num_devices: int = 1,
    num_generations: int = 7,
    user_content_field: str = 'question',
    tensor_parallel_size: int = 1,
    prefix_length: int = 768,
    generation_length: int = 4096,
    enforce_eager: bool = False,
    max_num_seqs: int = 512,
    dtype: str = 'bfloat16',
    return_raw_output: bool = False,
):

    tokenizer = AutoTokenizer.from_pretrained(model)

    def make_prompt(example):
        convo = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example[user_content_field]},
            ],
        }
        prompt = apply_chat_template(convo, tokenizer)
        prompt['prompt'] = prompt['prompt'] + "Let me solve this step by step.\n<think>"
        return prompt

    dataset = dataset.map(make_prompt)

    llm = LLM(
        model=model, 
        device=f'cuda', 
        dtype=dtype, 
        gpu_memory_utilization=0.95,
        max_num_seqs=max_num_seqs,
        hf_overrides = {
            'max_position_embeddings': prefix_length + generation_length
        },
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
    )

    # now GRPO has been refactored to have total number of samples
    # num_generations * num_devices, which means that now
    # each device will never process more than 1 dataset sample
    # at a time
    # https://github.com/huggingface/trl/pull/2776

    T = time.time()
    raw_output = llm.generate(
        [eg['prompt'] for eg in dataset.select(range(num_devices))], 
        SamplingParams(
            n=num_generations, 
            max_tokens=prefix_length + generation_length, 
            temperature=1,
        )
    )
    time_elapsed = time.time() - T

    examples = Dataset.from_list([
        {"input_ids": prompt.prompt_token_ids + list(completion.token_ids)} 
        for prompt in raw_output
        for completion in prompt.outputs
    ])

    times = [out.metrics.finished_time - out.metrics.arrival_time 
            for out in raw_output]
    metrics = {}
    metrics['time_taken_sum'] = sum(times)
    metrics['time_taken_mean'] = sum(times) / len(times)
    metrics['time_elapsed'] = time_elapsed
    metrics['model'] = model
    metrics['dtype'] = dtype
    metrics['max_num_seqs'] = max_num_seqs
    metrics['prefix_length'] = prefix_length
    metrics['generation_length'] = generation_length
    metrics['num_devices'] = num_devices
    metrics['num_generations'] = num_generations
    metrics['tensor_parallel_size'] = tensor_parallel_size
    metrics['enforce_eager'] = enforce_eager


    if return_raw_output:
        return (examples, metrics, raw_output)
    return (examples, metrics)


def estimate_tune(
    model: str,
    dataset: Dataset,
    accelerator: Accelerator,
    train_batch_size: int,
    dtype: str = 'bfloat16',
    use_flash_attn: bool = False,
    num_trials: int = 100,
):
    tokenizer = AutoTokenizer.from_pretrained(model)

    # test flattening?
    data_collator = DataCollatorWithPadding(tokenizer)

    # TODO: handle GA?
    # gradient_accumulation_steps = math.ceil(
    #     script_args.num_devices / script_args.train_batch_size
    # )
    dataloader = DataLoader(
        dataset, 
        batch_size=train_batch_size, 
        collate_fn=data_collator
    )

    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=getattr(torch, dtype),
        use_flash_attention_2=use_flash_attn, 
        low_cpu_mem_usage=True, # set this manually to also support torchrun
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    # activation checkpointing
    # NOTE: having some problems with no reentrant? check the loss
    # https://github.com/lessw2020/transformer_central/blob/main/activation_checkpointing_tutorial/activation_checkpointing_tutorial.ipynb
    from functools import partial
    check_fn = lambda submodule: submodule.__class__.__name__ in model._no_split_modules
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        # offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.REENTRANT,
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

    # NOTE: we are not making a seperate copy of the ref
    # at this point
    ref_model = model

    # but some bogus learning rate
    optimizer = AdamW(model.parameters(), lr=1e-6)

    optimizer = accelerator.prepare(optimizer)
    
    def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
            ).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]

            return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    times_taken = []
    for _ in trange(num_trials, disable=not accelerator.is_main_process):

        # this simulates the GA
        T = time.time()
        optimizer.zero_grad()

        loss = 0
        for batch in dataloader:

            # for reference
            # with torch.no_grad():
            #     model(**batch)

            # for the reference model
            # TODO: we are not making a seperate copy
            # of the reference model yet
            # assume the prompt length is about 100
            logits_to_keep = batch['input_ids'].size(1) - 100
            per_token_logps = get_per_token_logps(
                model, batch['input_ids'], None, logits_to_keep
            )
            with torch.inference_mode():
                ref_per_token_logps = get_per_token_logps(
                    ref_model, batch['input_ids'], None, logits_to_keep
                )

            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

            # skipped out the reward compute
            # fake the loss
            loss = loss + (
                torch.exp(per_token_logps - per_token_logps.detach()) + 
                per_token_kl
            )

            # actual training
            # batch['labels'] = batch['input_ids']
            # loss = model(**batch)
            # loss.backward()

        loss.sum(axis=1).mean().backward()
        optimizer.step()

        # to simulate passing off the VLLM
        # we dont measure the VLLM model loading times here
        torch.cuda.synchronize()
        T = time.time() - T
        times_taken.append(T)

    return times_taken

if __name__ == "__main__":

    parser = TrlParser(Arguments)
    script_args, = parser.parse_args_and_config()

    dataset = load_dataset(script_args.dataset_name, split='train')

    try:
        os.makedirs(script_args.results_dir)
    except FileExistsError:
        pass

    accelerator = Accelerator()
    timestamp = str(datetime.datetime.now()).replace(' ', '-')

    train_dataset_path = os.path.join(script_args.results_dir, DATASET_FILE)
    train_dataset_path = train_dataset_path.format(
        script_args.num_devices,
        script_args.num_generations,
    )
    if script_args.run_vllm and accelerator.is_main_process: 

        train_dataset, vllm_metrics = estimate_vllm(
            script_args.model_name,
            dataset,
            num_devices=script_args.num_devices,
            num_generations=script_args.num_generations,
            tensor_parallel_size=script_args.vllm_tensor_parallel_size,
        )
        train_dataset.to_json(train_dataset_path)

        with open(os.path.join(script_args.results_dir, METRICS_VLLM_FILE.format(timestamp)), 'w') as f:
            json.dump(vllm_metrics, f)

    # bench training
    if script_args.run_training:

        train_dataset = Dataset.from_json(train_dataset_path, split='train')

        times_taken = estimate_tune(
            script_args.model_name,
            train_dataset,
            accelerator,
            train_batch_size=script_args.train_batch_size,
            use_flash_attn=script_args.use_flash_attn,
        )

        if accelerator.is_main_process:
            with open(os.path.join(script_args.results_dir, METRICS_TRAIN_FILE.format(timestamp)), 'w') as f:
                json.dump({
                    'model_name': script_args.model_name,
                    'dataset_name': script_args.dataset_name,
                    'num_devices': script_args.num_devices,
                    'train_batch_size': script_args.train_batch_size, 
                    'use_flash_attn': script_args.use_flash_attn,
                    'time_taken': np.mean(times_taken)
                },f)

    
