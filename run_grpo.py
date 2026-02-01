import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vllm import LLM, SamplingParams

from algorithms import evaluate_vllm, load_model_and_dataset
from grpo import grpo_microbatch_train_step
from sft import get_response_log_probs, tokenize_prompt_and_output


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

def train_one_epoch(model, hyperparams):
    raise NotImplementedError

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
        dataset_name = hyperparams["dataset_name"].split(".")[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logdir = f"runs/{dataset_name}/{timestamp}"
        config_filename = f"{logdir}/config.json"
        os.makedirs(os.path.dirname(config_filename), exist_ok=True)
        with open(config_filename, "w") as json_file:
            json.dump(hyperparams, json_file, indent=4)
    else:
        logdir = args.load_checkpoint.rsplit("/", 1)[0]
        epoch_index = args.load_checkpoint.rsplit("/", 1)[1]
        hyperparams = read_json_to_dict(Path(f"{logdir}/config.json"))

    model = LLM(model=hyperparams["model_str"], dtype=hyperparams["dtype"])

    opt_params = hyperparams["optimizer_params"]
    optimizer = torch.optim.AdamW(model.params(),
                                  lr=opt_params["learning_rate"],
                                  betas=opt_params["betas"],
                                  weight_decay=opt_params["weight_decay"])