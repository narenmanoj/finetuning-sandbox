import argparse
from datetime import datetime
import gc
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from algorithms import evaluate_vllm, load_model_and_dataset
from grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from math_grader import r1_zero_reward_fn
from sft import get_response_log_probs, tokenize_prompt_and_output

EPOCH_KEY = "epoch"
MODEL_STATE_KEY = "model_state_dict"
OPTIMIZER_STATE_KEY = "optimizer_state_dict"
REWARD_KEY = "reward"


def read_json_to_dict(filename):
    """Reads a JSON file and returns a Python dictionary."""
    try:
        with open(filename, "r") as file_in:
            data_dict = json.load(file_in)
            return data_dict
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        current_directory = os.getcwd()
        print("Current Working Directory:", current_directory)
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{filename}'. Check file format.")
        return None

def train_one_epoch(model,
                    rollout_client,
                    optimizer,
                    tokenizer,
                    sampling_params_dict,
                    hyperparams,
                    epoch_index,
                    tb_writer,
                    reward_fn,
                    dataloader,
                    logdir,
                    device,
                    val_dataloader=None,
                    print_every=100):
    running_reward = 0.0
    last_reward = 0.0
    num_epochs = hyperparams["n_grpo_steps"]
    microbatch_size = hyperparams["train_batch_size"] // hyperparams["gradient_accumulation_steps"]
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=True)

    for i, data in pbar:
        prompts = data["problem"]  # list[str] length = batch_size
        answers = data["answer"]
        texts = rollout_client.generate(prompts, sampling_params_dict)
        texts_flattened = list(itertools.chain.from_iterable(texts))
        answers_flattened = [s for s in answers for _ in range(hyperparams["group_size"])]
        tokenized = tokenize_prompt_and_output(prompt_strs=texts_flattened, output_strs=answers_flattened, tokenizer=tokenizer)
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        # Do the actual GRPO logic here
        n_train_steps = hyperparams["epochs_per_rollout_batch"] * (len(texts_flattened) // hyperparams["train_batch_size"])
        for j in range(n_train_steps):
            macrobatch_start = hyperparams["train_batch_size"] * j
            macrobatch_end = hyperparams["train_batch_size"] * (j + 1)
            
            for k in range(microbatch_size):
                microbatch_start = macrobatch_start + microbatch_size * k * hyperparams["group_size"]
                microbatch_end = macrobatch_start + microbatch_size * (k + 1) * hyperparams["group_size"]
                texts_microbatch = texts_flattened[microbatch_start: microbatch_end]
                answers_microbatch = answers_flattened[microbatch_start: microbatch_end]
                breakpoint()
                old_log_probs_dict = get_response_log_probs(model=model,
                                                            input_ids=input_ids[microbatch_start: microbatch_end],
                                                            labels=labels[microbatch_start: microbatch_end],
                                                            return_token_entropy=True,
                                                            with_grad=False,
                                                            device=device)
                old_log_probs = old_log_probs_dict["log_probs"]
                rewards_dict = compute_group_normalized_rewards(reward_fn=reward_fn,
                                                                rollout_responses=texts_microbatch,
                                                                repeated_ground_truths=answers_microbatch,
                                                                group_size=hyperparams["group_size"],
                                                                advantage_eps=hyperparams["advantage_eps"],
                                                                normalize_by_std=hyperparams["use_std_normalization"])
                raw_rewards = rewards_dict[1]
                advantages = rewards_dict[0]
                
                breakpoint()
                log_probs_dict = get_response_log_probs(model=model,
                                                        input_ids=input_ids_microbatch,
                                                        labels=label_ids_microbatch,
                                                        return_token_entropy=True,
                                                        with_grad=True)
                policy_log_probs = log_probs_dict["log_probs"]
                loss_dict = grpo_microbatch_train_step(policy_log_probs=policy_log_probs,
                                                       response_mask=tokenized["response_mask"],
                                                       gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
                                                       loss_type=hyperparams["loss_type"],
                                                       raw_rewards=raw_rewards,
                                                       advantages=advantages,
                                                       old_log_probs=old_log_probs[microbatch_start: microbatch_end],
                                                       cliprange=hyperparams["cliprange"])
                breakpoint()

            optimizer.step()
            optimizer.zero_grad()
    torch.save({
        EPOCH_KEY: epoch_index,
        MODEL_STATE_KEY: model.state_dict(),
        OPTIMIZER_STATE_KEY: optimizer.state_dict(),
        REWARD_KEY: last_reward,
    }, f"{logdir}/{epoch_index}_checkpoint.tar")
    return last_reward


def _vllm_worker_loop(in_q, out_q, model_path_or_id: str, tokenizer_path: str, default_sampling_params: dict, vllm_dtype: str, gpu_mem_util: float):
    # Pin THIS process to physical GPU 1 (it becomes cuda:0 inside the process)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path_or_id,
        tokenizer=tokenizer_path,
        dtype=vllm_dtype,                 # "bfloat16" or "auto"
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem_util,
    )

    while True:
        msg = in_q.get()
        if msg is None:
            break

        cmd = msg["cmd"]

        if cmd == "generate":
            prompts = msg["prompts"]
            sp_dict = dict(default_sampling_params)
            sp_dict.update(msg.get("sampling_params", {}))

            sp = SamplingParams(**sp_dict)
            outputs = llm.generate(prompts, sp)
            # outputs: len=B; each has .outputs list len=n
            texts = [[o.text for o in req.outputs] for req in outputs]
            out_q.put({"ok": True, "texts": texts})

        elif cmd == "reload":
            # Rebuild engine from new HF-format checkpoint dir
            model_path_or_id = msg["model_path_or_id"]
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            llm = LLM(
                model=model_path_or_id,
                tokenizer=tokenizer_path,
                dtype=vllm_dtype,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_mem_util,
            )
            out_q.put({"ok": True})

        else:
            out_q.put({"ok": False, "error": f"unknown cmd: {cmd}"})


class RolloutClient:
    def __init__(self, model_path_or_id: str, tokenizer_path: str, default_sampling_params: dict, vllm_dtype: str="bfloat16", gpu_mem_util=0.90, gpu_id=1):
        ctx = mp.get_context("spawn")  # important with CUDA
        self.in_q = ctx.Queue(maxsize=2)
        self.out_q = ctx.Queue(maxsize=2)
        self.proc = ctx.Process(
            target=_vllm_worker_loop,
            args=(self.in_q, self.out_q, model_path_or_id, tokenizer_path, default_sampling_params, vllm_dtype, gpu_mem_util),
        )
        old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.proc.start()
        if old_cvd is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd


    def generate(self, prompts, sampling_params_dict):
        self.in_q.put({"cmd": "generate", "prompts": prompts, "sampling_params": sampling_params_dict})
        resp = self.out_q.get()
        if not resp["ok"]:
            raise RuntimeError(resp["error"])
        return resp["texts"]

    def reload(self, model_path_or_id: str):
        self.in_q.put({"cmd": "reload", "model_path_or_id": model_path_or_id})
        resp = self.out_q.get()
        if not resp["ok"]:
            raise RuntimeError(resp["error"])

    def close(self):
        self.in_q.put(None)
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=10)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    num_gpus = torch.cuda.device_count()
    use_vllm_rollouts = num_gpus >= 2
    assert use_vllm_rollouts, "This will not work properly with fewer than 2 GPUs"
    # from vllm import SamplingParams
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device = {device}")

    parser = argparse.ArgumentParser(description="A script that runs GRPO on Qwen with MATH dataset.")
    parser.add_argument("--load_checkpoint", type=str, help="The directory and epoch index from which to load. e.g. directory/5", default="")
    parser.add_argument("--config", type=str, help="Path to config file", default="")

    args = parser.parse_args()
    assert len(args.load_checkpoint) > 0 or len(args.config) > 0, "we need at least a checkpoint directory or a config file"
    assert len(args.load_checkpoint) == 0 or len(args.config) == 0, "only one of the checkpoint directory or the config file can be active"

    if len(args.config) > 0:
        hyperparams = read_json_to_dict(args.config)
        dataset_str = hyperparams["dataset_str"].split(".")[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logdir = f"runs/{dataset_str}/{timestamp}"
        config_filename = f"{logdir}/config.json"
        os.makedirs(os.path.dirname(config_filename), exist_ok=True)
        with open(config_filename, "w") as json_file:
            json.dump(hyperparams, json_file, indent=4)
    else:
        logdir = args.load_checkpoint.rsplit("/", 1)[0]
        epoch_index = args.load_checkpoint.rsplit("/", 1)[1]
        hyperparams = read_json_to_dict(Path(f"{logdir}/config.json"))

    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_str"], use_fast=True, fix_mistral_regex=True)
    model, train_dataset, test_dataset = load_model_and_dataset(model_str=hyperparams["model_str"],
                                                                dataset_str=hyperparams["dataset_str"],
                                                                prompt="prompts/r1_zero.prompt",
                                                                device=device,
                                                                dtype=hyperparams["dtype"])
    model.train()


    vllm_snapshot_dir = f"{logdir}/vllm_snapshot"
    os.makedirs(vllm_snapshot_dir, exist_ok=True)
    tokenizer.save_pretrained(vllm_snapshot_dir)
    model.save_pretrained(vllm_snapshot_dir, safe_serialization=True)
    default_sp = dict(
        temperature=hyperparams["sampling_temperature"],
        top_p=1.0,
        max_tokens=hyperparams["sampling_max_tokens"],
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=hyperparams["group_size"],
    )
    rollouts = RolloutClient(model_path_or_id=vllm_snapshot_dir,
                             tokenizer_path=vllm_snapshot_dir,
                             default_sampling_params=default_sp,
                             vllm_dtype=hyperparams["dtype"],
                             gpu_mem_util=hyperparams["gpu_memory_utilization"],
                             gpu_id=num_gpus - 1)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=hyperparams["rollout_batch_size"],
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=hyperparams["rollout_batch_size"],
                                 shuffle=True)
    opt_params = hyperparams["optimizer_params"]
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt_params["learning_rate"],
                                  betas=opt_params["betas"],
                                  weight_decay=opt_params["weight_decay"])
    
    current_epoch = 0
    if len(args.load_checkpoint) > 0:
        checkpoint_file = f"{logdir}/{epoch_index}_checkpoint.tar"
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint[MODEL_STATE_KEY])
        optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_KEY])
        current_epoch = checkpoint[EPOCH_KEY] + 1
        loaded_reward = checkpoint[REWARD_KEY]
        print(f"Resuming from beginning of epoch {current_epoch}")
        print(f"Current reward = {loaded_reward}")
    tb_writer = SummaryWriter(logdir)
    for epoch_it in tqdm(range(current_epoch, hyperparams["n_grpo_steps"], 1)):
        train_one_epoch(model=model,
                        rollout_client=rollouts,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        hyperparams=hyperparams,
                        sampling_params_dict=default_sp,
                        epoch_index=epoch_it,
                        tb_writer=tb_writer,
                        reward_fn=r1_zero_reward_fn,
                        dataloader=train_dataloader,
                        val_dataloader=test_dataloader,
                        device=device,
                        logdir=logdir,
        )
        # refresh vLLM weights every epoch step
        step_dir = f"{vllm_snapshot_dir}/step_{epoch_index}"
        os.makedirs(step_dir, exist_ok=True)
        model.save_pretrained(step_dir, safe_serialization=True)
        tokenizer.save_pretrained(step_dir, safe_serialization=True)
        rollouts.reload(step_dir)

    rollouts.close()
    del model
    gc.collect()