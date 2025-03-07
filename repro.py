import torch.distributed as dist
from datetime import timedelta
import os
import torch 
from transformers import AutoModelForCausalLM

PROMPTS = [
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nThree spheres with radii $11,$ $13,$ and $19$ are mutually externally tangent. A plane intersects the spheres in three congruent circles centered at $A,$ $B,$ and $C,$ respectively, and the centers of the spheres all lie on the same side of this plane. Suppose that $AB^2 = 560.$ Find $AC^2.$<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>',
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nPositive integers $a$ and $b$ satisfy the condition \\[\\log_2(\\log_{2^a}(\\log_{2^b}(2^{1000}))) = 0.\\] Find the sum of all possible values of $a+b$ .<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>',
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nRhombus $PQRS^{}_{}$ is inscribed in rectangle $ABCD^{}_{}$ so that vertices $P^{}_{}$ , $Q^{}_{}$ , $R^{}_{}$ , and $S^{}_{}$ are interior points on sides $\\overline{AB}$ , $\\overline{BC}$ , $\\overline{CD}$ , and $\\overline{DA}$ , respectively. It is given that $PB^{}_{}=15$ , $BQ^{}_{}=20$ , $PR^{}_{}=30$ , and $QS^{}_{}=40$ . Let $m/n^{}_{}$ , in lowest terms, denote the perimeter of $ABCD^{}_{}$ . Find $m+n^{}_{}$ .<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>',
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> and \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nFind the last three digits of the product of the positive roots of $\\sqrt{1995}x^{\\log_{1995}x}=x^2.$<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>'
]

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# to run, the 72B model required to trigger the issue is quite big. 
# - so use 4 gpus (we use 4 x A100_80g)
# TRANSFORMERS_VERBOSITY=error ACCELERATE_LOG_LEVEL=info accelerate launch \
#    --num_processes 4 repro.py

# printouts
# text (0):  We can visualize the spheres with radii 11, 13, and 19
# text (1): !!!!!!!!!!!!!!!!!!!!
# text (2): !!!!!!!!!!!!!!!!!!!!
# text (3): !!!!!!!!!!!!!!!!!!!!

def main(
    model_name = 'Qwen/Qwen2.5-Math-72B',
    model_in_train_name = 'facebook/opt-1.3b', # use a small model for demo
    gpu_memory_utilization = 0.5,
):

    # Enviromnent variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}'

    TP_SIZE = world_size
    number_of_steps = len(PROMPTS)

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=1800))
    llm = LLM(
        model=model_name,
        device='cuda',
        gpu_memory_utilization=gpu_memory_utilization,
        dtype='bfloat16',
        max_num_seqs=TP_SIZE,
        tensor_parallel_size=TP_SIZE,
        distributed_executor_backend="external_launcher",
        enforce_eager=True,
    ) 

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=20,
    )

    # training model
    if model_in_train_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_in_train_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_in_train_name, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.gradient_checkpointing_enable({"use_reentrant": False})
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )
        from functools import partial
        from accelerate.utils.dataclasses import get_module_class_from_name
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        transformer_cls_to_wrap = set()
        for layer_class in model._get_no_split_modules('cuda'):
            transformer_cls = get_module_class_from_name(model, layer_class)
            transformer_cls_to_wrap.add(transformer_cls)

        model = FSDP(
            model, 
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, 
                transformer_layer_cls=transformer_cls_to_wrap,
            ),
            sync_module_states=True,
            param_init_fn=lambda x: x.to_empty(device=device, recurse=False),
            device_id=device,
        )

        if local_rank == 0:
            print (model)

    for i in range(number_of_steps):
        outputs = llm.generate(
            [PROMPTS[i] for _ in range(TP_SIZE)],
            sampling_params=sampling_params, 
            use_tqdm=False,
        )
        # torch.distributed.breakpoint()
        output = outputs[0]
        text = output.outputs[0].text

        input_ids = torch.tensor(
            [tokenizer.encode(PROMPTS[i] + text)],
            device=device, dtype=torch.int32
        )
        if local_rank == 0:
            print (f"text ({i}):", text)
        if model_in_train_name:
            out = model(input_ids, labels=input_ids.long())
            loss = out.loss
            loss.backward()

if __name__ == '__main__':
    main()