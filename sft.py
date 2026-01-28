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
    stitched = [(torch.tensor(prompt_tokenized[i] + output_tokenized[i]), torch.tensor([0] * len(prompt_tokenized[i] + [1] * len(output_tokenized[i])))) for i in range(len(prompt_tokenized))]
    tokens = [stitch[0] for stitch in stitched]
    mask = [stitch[1] for stitch in stitched]
    stitched = pad_sequence(tokens, batch_first=True, padding_value=-1, padding_side="left")
    response = pad_sequence(mask, batch_first=True, padding_value=0, padding_side="left")
    breakpoint()
    ans = {
        "input_ids": torch.tensor(prompt_tokenized),
        "labels": torch.tensor(output_tokenized),
        "response_mask": response,
    }
    return ans