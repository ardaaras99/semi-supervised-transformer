# %%
import os
import warnings

import datasets
import torch

from src.dataset import DatasetArguments, GLUEDataset, TokenizerArguments
from src.engine import Trainer
from src.models import VGClassifier

warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_checkpoint = "distilbert-base-uncased"
batch_size = 32
task = "cola"
actual_task = "mnli" if task == "mnli-mm" else task

tokenizer_args = TokenizerArguments(model_name=model_checkpoint, max_length=128)
dataset_args = DatasetArguments(task=task, t_args=tokenizer_args, batch_size=batch_size)
glue_dataset = GLUEDataset(args=dataset_args)
metric = datasets.load_metric("glue", actual_task)

if task == "stsb":
    metric_name = "pearson"
elif task == "cola":
    metric_name = "matthews_correlation"
else:
    metric_name = "accuracy"

# model = BaseClassifier(model_checkpoint, 2)
model = VGClassifier(model_checkpoint, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

trainer = Trainer(model, optimizer, glue_dataset, metric, metric_name)
trainer.pipeline(max_epochs=5, patience=3, wandb_flag=False)


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     if task != "stsb":
#         predictions = np.argmax(predictions, axis=1)
#     else:
#         predictions = predictions[:, 0]
#     return metric.compute(predictions=predictions, references=labels)


# # validation_key = (
# #     "validation_mismatched"
# #     if task == "mnli-mm"
# #     else "validation_matched"
# #     if task == "mnli"
# #     else "validation"
# # )
