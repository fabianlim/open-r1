from ..grpo import (
    SYSTEM_PROMPT
)
from datasets import load_dataset
import re

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
except ImportError:
    pass


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

def _match_completions_for_format(completion_contents):
    pattern = r"^.*?</think>\s*<answer>\s*(.*?)\s*</answer>$"
    return [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]

# add the <think> that will be missing in the completion
def format_fixed_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions
    matches = _match_completions_for_format(completion_contents)
    return [1.0 if match else 0.0 for match in matches]

def accuracy_relaxed_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""

    try:
        contents = [completion[0]["content"] for completion in completions]
    except:
        contents = completions
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            match, = _match_completions_for_format([content])
    
            try:
                answer_parsed = match.group(1)
                reward = float(verify(answer_parsed, gold_parsed))
            except:
                reward = 0.
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

def update_reward_funcs(registry, script_args):
    registry['format_fixed'] = format_fixed_reward 
    registry['accuracy_relaxed'] = accuracy_relaxed_reward 
