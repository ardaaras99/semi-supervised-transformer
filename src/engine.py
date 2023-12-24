import copy
from typing import Union

import datasets
import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm

from src.dataset import GLUEDataset
from src.models import BaseClassifier, VGClassifier


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, task: str, metric: datasets.Metric):
    if task != "stsb":
        preds = preds.argmax(-1)
    else:
        preds = preds.squeeze()
    return metric.compute(predictions=preds, references=labels)


class Trainer:
    def __init__(
        self,
        model: Union[BaseClassifier, VGClassifier],
        optimizer: torch.optim.Optimizer,
        data: GLUEDataset,
        metric: datasets.Metric,
        metric_name: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.metric = metric
        self.metric_name = metric_name

        self.model = self.model.to("mps")

    def pipeline(self, max_epochs: int, patience: int, wandb_flag: bool = False):
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        t = tqdm(range(max_epochs))
        # best_model = None

        for _ in t:
            self.model.train()
            e_loss = self.train_epoch()
            train_metrics = self.evaluate(data_key="train")
            test_metrics = self.evaluate(data_key="validation")

            print(train_metrics, test_metrics)

            # epoch_wandb_log(e_loss, train_acc, test_acc, epoch) if wandb_flag else None
            # best_test_acc, best_model = early_stopping(test_acc, self.model, epoch)

            # t.set_description(self.get_info(e_loss, best_test_acc, train_acc, test_acc))
            t.set_description(f"Loss: {e_loss:.4f}")

            if early_stopping.early_stop:
                break

        # self.best_model = best_model
        # self.best_test_acc = best_test_acc

    def train_epoch(self):
        e_loss = 0
        for i_ids, a_mask, y in tqdm(self.data.loaders["train"], leave=False):
            i_ids, a_mask, y = move("mps", i_ids, a_mask, y)
            e_loss += self._run_batch(i_ids, a_mask, y)
        return e_loss

    def _run_batch(self, input_ids, attention_mask, y):
        self.optimizer.zero_grad()
        y_hat = self.model(input_ids, attention_mask)
        loss = F.nll_loss(y_hat, y.reshape(-1))
        loss.backward()
        self.optimizer.step()
        b_loss = loss.item()
        return b_loss

    def evaluate(self, data_key: str):
        self.model.eval()

        with torch.no_grad():
            y_pred, y = [], []
            for i_ids, a_mask, y_b in self.data.loaders[data_key]:
                i_ids, a_mask = move("mps", i_ids, a_mask)
                y_pred_b = self.model(i_ids, a_mask).to("cpu")
                y_pred.append(y_pred_b)
                y.append(y_b)

            y = torch.cat(y, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            metrics = compute_metrics(y_pred, y, self.data.args.task, self.metric)
            return metrics

    def get_info(self, e_loss, best_test_acc, train_acc, test_acc):
        return f"Loss: {e_loss:.4f}, Best Test Acc: {best_test_acc:.3f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}"


## Utility functions for Trainer
def epoch_wandb_log(loss, train_acc, test_acc, epoch):
    wandb.log(data={"train/loss": loss}, step=epoch)
    wandb.log(data={"train/train_acc": train_acc}, step=epoch)
    wandb.log(data={"test/test_accuracy": test_acc}, step=epoch)


def move(device, *tensors):
    moved_tensors = []
    for tensor in tensors:
        moved_tensor = tensor.to(device)
        moved_tensors.append(moved_tensor)
    return moved_tensors


class EarlyStopping:
    def __init__(self, patience: int, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_test_acc = 0
        self.best_model = None
        self.early_stop = False

    def __call__(self, test_acc: float, model: torch.nn.Module, epoch: int):
        if test_acc > self.best_test_acc:
            self.counter = 0
            self.best_test_acc = test_acc
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping at epoch {epoch}")

        return self.best_test_acc, self.best_model
