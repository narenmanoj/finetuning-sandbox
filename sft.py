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
    stitched = [(torch.tensor(prompt_tokenized[i] + output_tokenized[i]), 
                 torch.tensor([0] * len(prompt_tokenized[i]) + [1] * len(output_tokenized[i]))) 
                 for i in range(len(prompt_tokenized))]
    tokens = [stitch[0] for stitch in stitched]
    mask = [stitch[1] for stitch in stitched]

    inputs = [tokenized[:-1] for tokenized in tokens]
    labels_lst = [tokenized[1:] for tokenized in tokens]
    mask = [tokenized[1:] for tokenized in mask]

    input_ids = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.vocab_size, padding_side="right")
    labels = pad_sequence(labels_lst, batch_first=True, padding_value=tokenizer.vocab_size, padding_side="right")
    mask = pad_sequence(mask, batch_first=True, padding_value=0, padding_side="right")
    ans = {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": mask,
    }
    breakpoint()
    return ans