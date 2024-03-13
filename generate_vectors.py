"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Usage:
python generate_vectors.py --layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --use_base_model --model_size 7b --data_path datasets/test.json --dataset_name test
python generate_vectors.py --layers 39 --use_base_model --model_size 7b --data_path datasets/alpaca_data.json --dataset_name alpaca
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize_llama import tokenize_llama
from utils.helpers import make_tensor_save_suffix

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are a helpful assistant."
SAVE_VECTORS_PATH = "new_vectors_2"


class ComparisonDataset(Dataset):
    def __init__(self, data_path, system_prompt, token, model_name_path, use_chat):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat

    def prompt_to_tokens(self, instruction, input):
        tokens = tokenize_llama(
            self.tokenizer,
            self.system_prompt,
            [(instruction, input)],
            no_final_eos=True,
            chat_model=self.use_chat,
        )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        input = item["input"]
        p_tokens = self.prompt_to_tokens(instruction, None)
        # n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens


def generate_save_vectors(
    layers: List[int], use_base_model: bool, model_size: str, data_path: str, dataset_name: str
):
    """
    layers: list of layers to generate vectors for
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    """
    if not os.path.exists(SAVE_VECTORS_PATH):
        os.makedirs(SAVE_VECTORS_PATH)

    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, SYSTEM_PROMPT, size=model_size, use_chat=not use_base_model
    )
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    # neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        SYSTEM_PROMPT,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
    )
    tokenizer = AutoTokenizer.from_pretrained(
            model.model_name_path, token=HUGGINGFACE_TOKEN
        )
    for p_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        # print(p_tokens)
        # n_tokens = n_tokens.to(model.device)
        
        output = tokenizer.decode(p_tokens[0][1:])
        print(output)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -1, :].detach().cpu() # 获取最后一个token的activation
            pos_activations[layer].append(p_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        t.save(
            all_pos_layer,
            os.path.join(
                SAVE_VECTORS_PATH,
                f"vec_layer_{make_tensor_save_suffix(layer, model.model_name_path, dataset_name)}.pt",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--use_base_model", action="store_true", default=True)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--data_path", type=str, default="datasets/alpaca_data.json")
    parser.add_argument("--dataset_name", type=str, default="alpaca")

    args = parser.parse_args()
    generate_save_vectors(
        args.layers, args.use_base_model, args.model_size, args.data_path, args.dataset_name
    )
