from datasets import (
    Dataset,
    load_dataset,
)
from functools import partial
import gc
from math_grader import r1_zero_reward_fn
import pandas as pd
from torch.utils.data import DataLoader
from typing import Callable, List
from vllm import LLM, SamplingParams


def format_dataset(base_prompt, dataset):
    df = dataset.to_pandas()
    df["problem"] = df["problem"].map(lambda x: base_prompt.format(question=x))
    return Dataset.from_pandas(df)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_sampling_params: SamplingParams,
    prompts: List[str],
    answers: List[str] | None=None,
    print_convo: bool=False,
):
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    rewards = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        generated_text = outputs[i].outputs[0].text
        rewards.append(reward_fn(generated_text, answers[i]))
        rewards[-1].update({"prompt": prompt})
        if print_convo:
            print(f"\nPrompt: {prompt!r}, Generated text: {generated_text!r}\n")
    return pd.DataFrame(rewards)


if __name__ == "__main__":
    with open("prompts/r1_zero.prompt", "r") as f:
        base_prompt = f.read()
    llm = LLM(model="Qwen/Qwen2.5-1.5B", dtype="float16")
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
    reward_df = evaluate_vllm(llm,
                              r1_zero_reward_fn,
                              sampling_params,
                              list(test_math_dataset.to_pandas()["problem"]),
                              list(test_math_dataset.to_pandas()["answer"]))
    reward_df.to_csv("outputs/qwen_rewards_base.csv")
    breakpoint()
    del llm
    gc.collect()