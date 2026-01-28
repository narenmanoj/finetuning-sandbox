import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict

def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    prompt_tokenized = torch.tensor(tokenizer(prompt_strs)["input_ids"])
    output_tokenized = torch.tensor(tokenizer(output_strs)["input_ids"])
    breakpoint()
    ans = {
        "input_ids": torch.tensor(prompt_tokenized),
        "labels": torch.tensor(output_tokenized),
        "response_mask": None,
    }
    return ans