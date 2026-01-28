import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict

def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    prompt_tokenized = [tokenizer.tokenize(pro)[:-1] for pro in prompt_strs]
    output_tokenized = [tokenizer.tokenize(out)[:-1] for out in output_strs]
    breakpoint()
    ans = {
        "input_ids": None,
        "labels": None,
        "response_mask": None,
    }
    return ans