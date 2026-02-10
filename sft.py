import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    # Ensure we have a valid pad token id
    if tokenizer.pad_token_id is None:
        # common safe fallback for causal LMs
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    prompt_tokenized = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_tokenized = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    stitched_tokens = []
    stitched_mask = []
    for p_ids, o_ids in zip(prompt_tokenized, output_tokenized):
        ids = torch.tensor(p_ids + o_ids, dtype=torch.long)
        m   = torch.tensor([0] * len(p_ids) + [1] * len(o_ids), dtype=torch.long)
        stitched_tokens.append(ids)
        stitched_mask.append(m)

    # Standard next-token LM shift:
    # input_ids = tokens[:-1], labels = tokens[1:], response_mask aligned with labels
    maxlen = max(len(x) for x in stitched_tokens) - 1
    inputs = [t[:maxlen] for t in stitched_tokens]
    labels = [t[1:maxlen+1] for t in stitched_tokens]
    resp_m = [m[1:maxlen+1] for m in stitched_mask]

    input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    labels    = pad_sequence(labels,  batch_first=True, padding_value=pad_id)
    resp_m    = pad_sequence(resp_m,  batch_first=True, padding_value=0)

    attention_mask = (input_ids != pad_id).to(torch.long)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": resp_m,        # 1 where label is response token
        "attention_mask": attention_mask
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    denom = torch.logsumexp(logits, dim=-1)
    logp = logits - denom.unsqueeze(-1)
    return -torch.sum(p * logp, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    return_token_entropy: bool = False,
    with_grad: bool = True,
    device=None,
) -> dict[str, torch.Tensor]:
    ctx_mgr = torch.enable_grad if with_grad else torch.no_grad
    with ctx_mgr():
        logits = model(input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        token_entropy = compute_entropy(logits) if return_token_entropy else None
        logp = logits - torch.logsumexp(logits, dim=-1).unsqueeze(-1)
        log_probs = torch.gather(logp, dim=-1, index=labels.to(device).unsqueeze(-1)).squeeze(-1)
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
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant=normalize_constant) / (gradient_accumulation_steps + 2)
    loss.backward()
    return (loss.detach(), {"Loss value": loss.detach()})