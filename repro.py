import torch.distributed as dist
from datetime import timedelta
import os
import torch 
from transformers import AutoModelForCausalLM
from tqdm import trange

# PROMPT_IDS = [
#     [151644, 8948, 198, 32, 10435, 1948, 2657, 323, 21388, 13, 576, 1196, 17064, 264, 3405, 11, 323, 279, 21388, 67477, 432, 13, 576, 17847, 1156, 15482, 911, 279, 32711, 1882, 304, 279, 3971, 323, 1221, 5707, 279, 1196, 448, 279, 4226, 13, 576, 32711, 1882, 374, 43810, 2878, 366, 26865, 29, 690, 26865, 29, 323, 279, 4226, 374, 2661, 304, 279, 1124, 79075, 4573, 11, 15576, 11, 600, 1734, 2572, 366, 26865, 29, 32711, 1882, 1588, 690, 26865, 29, 1124, 79075, 90, 9217, 1588, 7810, 151645, 198, 151644, 872, 198, 19641, 65718, 448, 11900, 72, 400, 16, 16, 4779, 400, 16, 18, 4779, 323, 400, 16, 24, 3 , 525, 52479, 68342, 68660, 13, 362, 11031, 88184, 279, 65718, 304, 2326, 30169, 11680, 25362, 30188, 518, 400, 32, 4779, 400, 33, 4779, 323, 400, 34, 4779, 15576, 11, 323, 279, 18652, 315, 279, 65718, 678, 10246, 389, 279, 1852, 3108, 315, 419, 11031, 13, 82610, 429, 400, 1867, 61, 17, 284, 220, 20, 21, 15, 2418, 7379, 400, 1706, 61, 17, 2418, 151645, 198, 151644, 77091, 198, 10061, 752, 11625, 419, 3019, 553, 3019, 624, 13708, 766, 29],
#     [151644, 8948, 198, 32, 10435, 1948, 2657, 323, 21388, 13, 576, 1196, 17064, 264, 3405, 11, 323, 279, 21388, 67477, 432, 13, 576, 17847, 1156, 15482, 911, 279, 32711, 1882, 304, 279, 3971, 323, 1221, 5707, 279, 1196, 448, 279, 4226, 13, 576, 32711, 1882, 374, 43810, 2878, 366, 26865, 29, 690, 26865, 29, 323, 279, 4226, 374, 2661, 304, 279, 1124, 79075, 4573, 11, 15576, 11, 600, 1734, 2572, 366, 26865, 29, 32711, 1882, 1588, 690, 26865, 29, 1124, 79075, 90, 9217, 1588, 7810, 151645, 198, 151644, 872, 198, 35490, 25780, 400, 64, 3, 323, 400, 65, 3, 26553, 279, 2971, 1124, 26056, 839, 62, 17, 11520, 839, 15159, 17, 61, 64, 92, 11520, 839, 15159, 17, 61, 65, 25547, 17, 47822, 16, 15, 15, 15, 92, 7705, 284, 220, 15, 7110, 60, 7379, 279, 2629, 315, 678, 3204, 2750, 315, 400, 64, 35093, 3, 659, 151645, 198, 151644, 77091, 198, 10061, 752, 11625, 419, 3019, 553, 3019, 624, 13708, 766, 29],
#     [151644, 8948, 198, 32, 10435, 1948, 2657, 323, 21388, 13, 576, 1196, 17064, 264, 3405, 11, 323, 279, 21388, 67477, 432, 13, 576, 17847, 1156, 15482, 911, 279, 32711, 1882, 304, 279, 3971, 323, 1221, 5707, 279, 1196, 448, 279, 4226, 13, 576, 32711, 1882, 374, 43810, 2878, 366, 26865, 29, 690, 26865, 29, 323, 279, 4226, 374, 2661, 304, 279, 1124, 79075, 4573, 11, 15576, 11, 600, 1734, 2572, 366, 26865, 29, 32711, 1882, 1588, 690, 26865, 29, 1124, 79075, 90, 9217, 1588, 7810, 151645, 198, 151644, 872, 198, 72162, 2855, 355, 400, 47, 90506, 61, 65797, 6257, 3, 374, 1640, 17433, 304, 22756, 400, 1867, 6484, 61, 65797, 6257, 3, 773, 429, 17228, 400, 47, 61, 65797, 6257, 3, 1154, 400, 48, 61, 65797, 6257, 3, 1154, 400, 49, 61, 65797, 6257, 3, 1154, 323, 400, 50, 61, 65797, 6257, 3, 525, 14791, 3501, 389, 11067, 57960, 1975, 1056, 90, 1867, 31716, 1154, 57960, 1975, 1056, 90, 4897, 31716, 1154, 57960, 1975, 1056, 90, 6484, 31716, 1154, 323, 57960, 1975, 1056, 90, 6352, 31716, 1154, 15576, 13, 1084, 374, 2661, 429, 400, 40637, 61, 65797, 6257, 28, 16, 20, 3, 1154, 400, 33, 48, 61, 65797, 6257, 28, 17, 15, 3, 1154, 400, 6480, 61, 65797, 6257, 28, 18, 15, 3, 1154, 323, 400, 70810, 61, 65797, 6257, 28, 19, 15, 3, 659, 6771, 400, 76, 9612, 61, 65797, 6257, 3, 1154, 304, 15457, 3793, 11, 78064, 279, 46342, 315, 400, 1867, 6484, 61, 65797, 6257, 3, 659, 7379, 400, 76, 38334, 61, 65797, 6257, 3, 659, 151645, 198, 151644, 77091, 198, 10061, 752, 11625, 419, 3019, 553, 3019, 624, 13708, 766, 29],
#     [151644, 8948, 198, 32, 10435, 1948, 2657, 323, 21388, 13, 576, 1196, 17064, 264, 3405, 11, 323, 279, 21388, 67477, 432, 13, 576, 17847, 1156, 15482, 911, 279, 32711, 1882, 304, 279, 3971, 323, 1221, 5707, 279, 1196, 448, 279, 4226, 13, 576, 32711, 1882, 374, 43810, 2878, 366, 26865, 29, 690, 26865, 29, 323, 279, 4226, 374, 2661, 304, 279, 1124, 79075, 4573, 11, 15576, 11, 600, 1734, 2572, 366, 26865, 29, 32711, 1882, 1588, 690, 26865, 29, 1124, 79075, 90, 9217, 1588, 7810, 151645, 198, 151644, 872, 198, 72162, 2855, 355, 400, 47, 90506, 61, 65797, 6257, 3, 374, 1640, 17433, 304, 22756, 400, 1867, 6484, 61, 65797, 6257, 3, 773, 429, 17228, 400, 47, 61, 65797, 6257, 3, 1154, 400, 48, 61, 65797, 6257, 3, 1154, 400, 49, 61, 65797, 6257, 3, 1154, 323, 400, 50, 61, 65797, 6257, 3, 525, 14791, 3501, 389, 11067, 57960, 1975, 1056, 90, 1867, 31716, 1154, 57960, 1975, 1056, 90, 4897, 31716, 1154, 57960, 1975, 1056, 90, 6484, 31716, 1154, 323, 57960, 1975, 1056, 90, 6352, 31716, 1154, 15576, 13, 1084, 374, 2661, 429, 400, 40637, 61, 65797, 6257, 28, 16, 20, 3, 1154, 400, 33, 48, 61, 65797, 6257, 28, 17, 15, 3, 1154, 400, 6480, 61, 65797, 6257, 28, 18, 15, 3, 1154, 323, 400, 70810, 61, 65797, 6257, 28, 19, 15, 3, 659, 6771, 400, 76, 9612, 61, 65797, 6257, 3, 1154, 304, 15457, 3793, 11, 78064, 279, 46342, 315, 400, 1867, 6484, 61, 65797, 6257, 3, 659, 7379, 400, 76, 38334, 61, 65797, 6257, 3, 659, 151645, 198, 151644, 77091, 198, 10061, 752, 11625, 419, 3019, 553, 3019, 624, 13708, 766, 29] 
# ]

PROMPTS = [
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nThree spheres with radii $11,$ $13,$ and $19$ are mutually externally tangent. A plane intersects the spheres in three congruent circles centered at $A,$ $B,$ and $C,$ respectively, and the centers of the spheres all lie on the same side of this plane. Suppose that $AB^2 = 560.$ Find $AC^2.$<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>',
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nPositive integers $a$ and $b$ satisfy the condition \\[\\log_2(\\log_{2^a}(\\log_{2^b}(2^{1000}))) = 0.\\] Find the sum of all possible values of $a+b$ .<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>',
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nRhombus $PQRS^{}_{}$ is inscribed in rectangle $ABCD^{}_{}$ so that vertices $P^{}_{}$ , $Q^{}_{}$ , $R^{}_{}$ , and $S^{}_{}$ are interior points on sides $\\overline{AB}$ , $\\overline{BC}$ , $\\overline{CD}$ , and $\\overline{DA}$ , respectively. It is given that $PB^{}_{}=15$ , $BQ^{}_{}=20$ , $PR^{}_{}=30$ , and $QS^{}_{}=40$ . Let $m/n^{}_{}$ , in lowest terms, denote the perimeter of $ABCD^{}_{}$ . Find $m+n^{}_{}$ .<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>',
 '<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and the answer is given in the \\boxed environment, respectively, i.e., <think> reasoning process here </think> and \\boxed{answer here}.<|im_end|>\n<|im_start|>user\nFind the last three digits of the product of the positive roots of $\\sqrt{1995}x^{\\log_{1995}x}=x^2.$<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>'
]

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# to run, the 72B model required to trigger the issue is quite big. 
# - so use 4 gpus (we use 4 x A100_80g)
# TRANSFORMERS_VERBOSITY=error ACCELERATE_LOG_LEVEL=info accelerate launch \
#    --num_processes 4 repro.py

# printouts
# text  We are given three spheres with radii 11, 13, and 19
# text !!!!!!!!!!!!!!!!!!!!
# text !!!!!!!!!!!!!!!!!!!!
# text !!!!!!!!!!!!!!!!!!!!

# if model_in_train_name is None, then we will get actual responses
# text  We are given three spheres with radii 11, 13, and 19
# text  First, let's unpack the expression $\log_2(\log_{2^a}(\log
# \begin{itemize}
# \item $PB = 15$
# \item $
# text  -Rhombus $PQRS$ inscribed in rectangle $ABCD$ with vertices $P

def main(
    model_name = 'Qwen/Qwen2.5-Math-72B',
    # model_in_train_name = 'Qwen/Qwen2.5-Math-7B', # just use a small model to trigger bug
    model_in_train_name = 'facebook/opt-1.3b', # use a small model for demo
    transformer_layer_cls=Qwen2DecoderLayer, # for FSDP wrapping of model in train
    max_prompt_length = 2048,
    max_completion_length = 768,
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
        # max_model_len=max_prompt_length + max_completion_length,
        # hf_overrides = {
        #     'max_position_embeddings': max_prompt_length + max_completion_length,
        # },
        max_num_seqs=TP_SIZE,
        # enable_prefix_caching=True,
        tensor_parallel_size=TP_SIZE,
        distributed_executor_backend="external_launcher",
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
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial
        model = FSDP(
            model, 
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, 
                transformer_layer_cls=(transformer_layer_cls,)
            ),
            sync_module_states=True,
            param_init_fn=lambda x: x.to_empty(device=device, recurse=False),
            device_id=device,
        )

        if local_rank == 0:
            print (model)

    for i in trange(number_of_steps, disable=rank>0):
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
            print ("text", text)
        if model_in_train_name:
            out = model(input_ids, labels=input_ids.long())
            loss = out.loss
            loss.backward()

if __name__ == '__main__':
    main()