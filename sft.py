import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from typing import List, Dict

def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    prompt_tokenized = tokenizer(prompt_strs)["input_ids"]
    output_tokenized = tokenizer(output_strs)["input_ids"]
    stitched = [torch.tensor(prompt_tokenized[i] + output_tokenized[i]) for i in range(len(prompt_tokenized))]
    stitched = pad_sequence(stitched, batch_first=True, padding_value=0)
    breakpoint()
    ans = {
        "input_ids": torch.tensor(prompt_tokenized),
        "labels": torch.tensor(output_tokenized),
        "response_mask": None,
    }
    return ans