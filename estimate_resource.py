from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
from trl import ScriptArguments, TrlParser
from trl.data_utils import apply_chat_template
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from accelerate import Accelerator

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

DATASET_FILE = 'dataset.json'
METRICS_VLLM_FILE = 'metrics_vllm.json'
METRICS_TRAIN_FILE = 'metrics_train.json'

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
        gpu_memory_utilization=0.8,
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

    raw_output = llm.generate(
        [eg['prompt'] for eg in dataset.select(range(num_devices))], 
        SamplingParams(
            n=num_generations, 
            max_tokens=prefix_length + generation_length, 
            temperature=1,
        )
    )

    examples = Dataset.from_list([
        {"input_ids": prompt.prompt_token_ids + list(completion.token_ids)} 
        for prompt in raw_output
        for completion in prompt.outputs
    ])

    metrics = {}
    metrics['time_taken'] = sum(
        [out.metrics.finished_time - out.metrics.arrival_time 
             for out in raw_output]) 
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
    # test flattening?
    data_collator = DataCollatorForSeq2Seq()

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

    # but some bogus learning rate
    optimizer = AdamW(model.parameters(), lr=1e-6)

    optimizer = accelerator.prepare(optimizer)

    times_taken = []
    for _ in trange(num_trials, disable=not accelerator.is_main_process):

        # this simulates the GA
        T = time.time()
        optimizer.zero()
        for batch in dataloader:
        # for __ in range(gradient_accumulation_steps):

            # for reference
            with torch.no_grad():
                model(**batch)

            # actual training
            batch['labels'] = batch['input_ids']
            loss = model(**batch)
            loss.backward()

        optimizer.step()

        # to simulate passing off the VLLM
        # we dont measure the VLLM model loading times here
        torch.cuda.synchronize()
        T = time.tim() - T
        times_taken.append(T)

    return times_taken

if __name__ == "__main__":

    parser = TrlParser(Arguments)
    script_args = parser.parse_args_and_config()

    dataset = load_dataset(script_args.dataset_name, split='train')

    accelerator = Accelerator()

    train_dataset_path = os.path.join(script_args.results_dir, DATASET_FILE)
    if script_args.run_vllm and accelerator.is_main_process: 

        train_dataset, vllm_metrics = estimate_vllm(
            script_args.model_name,
            dataset,
            num_devices=script_args.num_devices,
            num_generations=script_args.num_generations,
            tensor_parallel_size=script_args.vllm_tensor_parallel_size,
        )
        train_dataset.to_json(train_dataset_path)

        with open(os.path.join(script_args.results_dir, METRICS_TRAIN_FILE)) as f:
            json.dump(vllm_metrics)
    else:
        train_dataset = Dataset.from_json(train_dataset_path, split='train')

    # bench training

    if script_args.run_training:
        times_taken = estimate_tune(
            script_args.model_name,
            train_dataset,
            train_batch_size=script_args.train_batch_size,
            use_flash_attn=script_args.use_flash_attn,
        )

        if accelerator.is_main_process:
            with open(os.path.join(script_args.results_dir, METRICS_TRAIN_FILE)) as f:
                json.dump({
                    'model_name': script_args.model_name,
                    'dataset_name': script_args.dataset_name,
                    'num_devices': script_args.num_devices,
                    'train_batch_size': script_args.train_batch_size, 
                    'use_flash_attn': script_args.use_flash_attn,
                    'time_taken': np.mean(times_taken)
                },f)

    
