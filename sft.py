import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict

def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    ans = {
        "input_ids": None,
        "labels": None,
        "response_mask": None,
    }
    return ans