from datasets import (
    Dataset,
    load_dataset,
)
from functools import partial
from torch.utils.data import DataLoader
from typing import Callable, List
from vllm import LLM, SamplingParams

from math_grader import r1_zero_reward_fn

def format_dataset(base_prompt, dataset):
    df = dataset.to_pandas()
    df["problem"] = df["problem"].map(lambda x: base_prompt.format(question=x))
    return Dataset.from_pandas(df)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
):
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    raise NotImplementedError


if __name__ == "__main__":
    with open("prompts/r1_zero.prompt", "r") as f:
        base_prompt = f.read()
    # llm = LLM(model="Qwen/Qwen2.5-1.5B")
    math_dataset = load_dataset("hiyouga/math12k")
    train_math_dataset = format_dataset(base_prompt, math_dataset["train"])
    test_math_dataset = format_dataset(base_prompt, math_dataset["test"])
    train_dataloader = DataLoader(train_math_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_math_dataset, batch_size=8, shuffle=True)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )