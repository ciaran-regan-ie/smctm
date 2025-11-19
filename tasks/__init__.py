from omegaconf import DictConfig

from tasks.task import Task
from tasks.few_shot_image_classification import FewShotImageClassificationTask

__all__ = ["FewShotImageClassification"]

_TASKS = {"FewShotImageClassification": FewShotImageClassificationTask}

# Constructs a task based on the config
def construct_task(cfg: DictConfig, logger=None) -> Task:
	return _TASKS[cfg.task.type](cfg, logger=logger)