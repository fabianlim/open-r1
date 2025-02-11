from ..grpo import (
    reward_funcs_registry, SYSTEM_PROMPT
)
from datasets import load_dataset

def make_prefix(dp):
    # """This works for any base model"""
    prefix = f"""{SYSTEM_PROMPT}.
User: Help me solve the following math question: {dp['question']}.
Assistant: Let me solve this step by step.
<think>"""
    return prefix

def make_conversation(example):
    # for now.. only support base model with no conversation
    return {
        "prompt": make_prefix(example)
    }
