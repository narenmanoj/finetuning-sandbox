import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict

def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    prompt_tokenized = tokenizer(prompt_strs)["input_ids"]
    output_tokenized = tokenizer(output_strs)["input_ids"]
    stitched = [prompt_tokenized[i] + output_tokenized[i] for i in range(len(prompt_tokenized))]
    breakpoint()
    ans = {
        "input_ids": torch.tensor(prompt_tokenized),
        "labels": torch.tensor(output_tokenized),
        "response_mask": None,
    }
    return ans