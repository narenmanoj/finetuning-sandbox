from datasets import (
    Dataset,
    load_dataset,
)
import gc
from math_grader import r1_zero_reward_fn
import pandas as pd
from transformers import AutoModelForCausalLM
import torch
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
):
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    def _score_output(prompt, output, answer):
        generated_text = output.outputs[0].text
        rw = reward_fn(generated_text, answer)
        rw.update({"prompt": prompt, "generated_text": generated_text})
        return rw
    rewards = [_score_output(prompts[i], outputs[i], answers[i]) for i in range(len(prompts))]
    return pd.DataFrame(rewards)


def load_model_and_dataset(model_str, dataset_str, prompt=None, dtype="float16", device=None):
    base_prompt = None
    if prompt is not None:
        with open(prompt, "r") as f:
            base_prompt = f.read()
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=dtype,
        device_map=device,
    )
    math_dataset = load_dataset(dataset_str)
    train_dataset = format_dataset(base_prompt, math_dataset["train"])
    test_dataset = format_dataset(base_prompt, math_dataset["test"])
    
    return (model, train_dataset, test_dataset)


if __name__ == "__main__":
    llm, train_dataset, test_dataset = load_model_and_dataset(model_str="Qwen/Qwen2.5-Math-1.5B",
                                                              dataset_str="hiyouga/math12k",
                                                              prompt="prompts/r1_zero.prompt",
                                                              dtype="float16")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
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
                              list(test_dataset.to_pandas()["problem"]),
                              list(test_dataset.to_pandas()["answer"]))
    reward_df.to_csv("outputs/qwen_rewards_base.csv")
    del llm
    gc.collect()