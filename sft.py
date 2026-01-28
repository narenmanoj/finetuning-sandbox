import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer
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

    maxlen = max([len(x) for x in tokens]) - 1
    inputs = [tokenized[:min(maxlen, len(tokenized))] for tokenized in tokens]
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
    return ans


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    denom = torch.logsumexp(logits, dim=-1)
    logp = logits - denom.unsqueeze(-1)
    return -torch.sum(p * logp, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    token_entropy = compute_entropy(logits) if return_token_entropy else None
    logp = logits - torch.logsumexp(logits, dim=-1).unsqueeze(-1)
    log_probs = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return {"log_probs": log_probs,
            "token_entropy": token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    tensor_masked = tensor.masked_fill(~mask, 0.0)
    return torch.sum(tensor_masked, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant) / gradient_accumulation_steps
    loss.backward()
    return loss