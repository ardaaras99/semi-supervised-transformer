from dataclasses import dataclass

import datasets
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.constants import TASK_TO_KEYS


@dataclass
class TokenizerArguments:
    model_name: str
    max_length: int
    padding: bool = True
    truncation: bool = True
    use_fast: bool = True


@dataclass
class DatasetArguments:
    task: str
    t_args: TokenizerArguments
    batch_size: int


class GLUEDataset:
    def __init__(self, args: DatasetArguments):
        self.args = args
        self.key1, self.key2 = TASK_TO_KEYS[args.task]
        self.tokenizer = AutoTokenizer.from_pretrained(args.t_args.model_name, use_fast=args.t_args.use_fast)
        self.raw_data = datasets.load_dataset("glue", args.task)
        self.tensor_data, self.loaders = {}, {}
        for split in ["train", "validation"]:
            tmp = self._processes_data(self.raw_data[split])
            i_ids, a_mask = (tmp["input_ids"], tmp["attention_mask"])
            labels = torch.tensor(
                self.raw_data[split]["label"],
                dtype=torch.long,
            ).reshape(-1, 1)
            self.tensor_data[split] = TensorDataset(i_ids, a_mask, labels)
            self.loaders[split] = DataLoader(
                self.tensor_data[split],
                batch_size=args.batch_size,
                shuffle=True,
            )

    def _processes_data(self, texts):
        if self.key2 is None:
            return self.tokenizer(
                texts[self.key1],
                truncation=self.args.t_args.truncation,
                padding=self.args.t_args.padding,
                max_length=self.args.t_args.max_length,
                return_tensors="pt",
            )
        else:
            return self.tokenizer(
                texts[self.key1],
                texts[self.key2],
                truncation=self.args.t_args.truncation,
                padding=self.args.t_args.padding,
                max_length=self.args.t_args.max_length,
                return_tensors="pt",
            )
