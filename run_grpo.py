import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from algorithms import evaluate_vllm, load_model_and_dataset
from grpo import grpo_microbatch_train_step
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
                    optimizer,
                    tokenizer,
                    sampling_params,
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
    gradient_accumulation_steps = hyperparams["gradient_accumulation_steps"]
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=True)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in pbar:
        breakpoint()
    torch.save({
        EPOCH_KEY: epoch_index,
        MODEL_STATE_KEY: model.state_dict(),
        OPTIMIZER_STATE_KEY: optimizer.state_dict(),
        REWARD_KEY: last_reward,
    }, f"{logdir}/{epoch_index}_checkpoint.tar")
    return last_reward

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_str"], use_fast=True)
    model, train_dataset, test_dataset = load_model_and_dataset(model_str=hyperparams["model_str"],
                                                                dataset_str=hyperparams["dataset_str"],
                                                                prompt="prompts/r1_zero.prompt",
                                                                device=device,
                                                                dtype=hyperparams["dtype"])
    model.train()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=hyperparams["train_batch_size"],
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=hyperparams["train_batch_size"],
                                 shuffle=True)
    opt_params = hyperparams["optimizer_params"]
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt_params["learning_rate"],
                                  betas=opt_params["betas"],
                                  weight_decay=opt_params["weight_decay"])
    sampling_params = SamplingParams(
        temperature=hyperparams["sampling_temperature"],
        top_p=1.0,
        max_tokens=hyperparams["sampling_max_tokens"],
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
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
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        sampling_params=sampling_params,
                        hyperparams=hyperparams,
                        epoch_index=epoch_it,
                        tb_writer=tb_writer,
                        reward_fn=r1_zero_reward_fn,
                        dataloader=train_dataloader,
                        val_dataloader=test_dataloader,
        )
    del model
    gc.collect()