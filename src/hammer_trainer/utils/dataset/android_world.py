from torch.utils.data import Dataset
from typing import Any, Dict, List

import json
import logging
import os
import uuid

from hammer_server.utils import get_task_list

logger = logging.getLogger(__file__)


class AndroidWorldDataset(Dataset):
    def __init__(self, filepath=None, dataset_size: int = None):
        tasks = None
        data_dir = None
        if filepath is not None:
            if not os.path.isfile(filepath):
                logger.warning(f"{filepath} is not a file")
            else:
                with open(file=filepath, mode="r", encoding="utf-8") as f:
                    tasks = json.load(f)
                data_dir = os.path.split(filepath)[0]
        self.tasks = tasks or get_task_list()
        self.data_dir = data_dir
        self.dataset_size = (
            len(self.tasks) if dataset_size is None else min(len(self.tasks), dataset_size)
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        task = self.tasks[index]
        if isinstance(task, dict):
            params = os.path.join(self.data_dir, f"""{task["task_id"]}.pkl""")
            task["params"] = params
            return task
        else:
            return {"task_id": f"{task}_{uuid.uuid4().hex}", "task": task}
