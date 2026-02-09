from einops import einsum
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Callable, List, Literal, Dict

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    rollout_batch_size = len(rollout_responses)
    rewards = [reward_fn(rollout_responses[i], repeated_ground_truths[i]) for i in range(rollout_batch_size)]
    rewards_df = pd.DataFrame(rewards)
    rewards_tensor = torch.tensor(rewards_df["reward"]).reshape((rollout_batch_size // group_size, group_size))
    ans = rewards_tensor - torch.mean(rewards_tensor, dim=1).unsqueeze(-1)
    if normalize_by_std:
        ans = ans / (torch.std(ans, dim=1).unsqueeze(-1) + advantage_eps)
    return (ans.flatten(), rewards_tensor.flatten(), {"rewards_df": rewards_df})


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -einsum(raw_rewards_or_advantages, policy_log_probs, "batch scalar, batch sequence -> batch sequence")


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def _apply_adv(adv, impo):
        return einsum(adv, impo, "batch scalar, batch sequence -> batch sequence")
    importances = torch.exp(policy_log_probs - old_log_probs)
    t1 = _apply_adv(advantages, importances)
    importances_clip = torch.clip(importances, min=1 - cliprange, max=1 + cliprange)
    t1_clip = _apply_adv(advantages, importances_clip)
    return (-torch.minimum(t1, t1_clip), {"metadata": 0})


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
    if loss_type == "reinforce_with_baseline":
        raw_rewards_or_advantages = advantages
    elif loss_type == "no_baseline":
        raw_rewards_or_advantages = raw_rewards
    ans = compute_naive_policy_gradient_loss(raw_rewards_or_advantages=raw_rewards_or_advantages,
                                             policy_log_probs=policy_log_probs)
    return (ans, {"metadata": 0})


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(torch.bool)
    return torch.nanmean(torch.masked_fill(tensor, ~mask, float("nan")), dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None, 
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_dict = compute_policy_gradient_loss(policy_log_probs=policy_log_probs,
                                             loss_type=loss_type,
                                             raw_rewards=raw_rewards,
                                             advantages=advantages,
                                             old_log_probs=old_log_probs,
                                             cliprange=cliprange)
    loss = loss_dict[0]
    loss = masked_mean(loss, response_mask, dim=1)
    loss = torch.mean(loss)
    loss /= gradient_accumulation_steps
    loss.backward()
    return (loss, loss_dict[1])