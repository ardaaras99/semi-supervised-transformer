GLUE_TASKS = [
    "cola",
    # "mnli",
    # "mnli-mm",
    "mrpc",
    # "qnli",
    # "qqp",
    "rte",
    # "sst2",
    "stsb",
    "wnli",
]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
