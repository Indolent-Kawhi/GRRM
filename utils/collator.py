import torch
import copy
import argparse
from dataclasses import dataclass

from torch.utils.data import Sampler
import torch.distributed as dist
from data_process.prompt import SFT_SYSTEM_PROMPT


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text = full_texts,
            text_target = input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels


        return inputs



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        self.tokenizer.padding_side = "left"

    def __call__(self, batch):

        input_texts = [
            self.tokenizer.apply_chat_template(
                [
                    # {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": d["input_ids"]}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for d in batch
        ]

        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        return (inputs, targets)
    
class TestRLCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        self.tokenizer.padding_side = "left"

    def __call__(self, batch):

        input_texts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": d["input_ids"]}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for d in batch
        ]

        input_texts = [input_text+"<think>\n" for input_text in input_texts]

        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        return (inputs, targets)

