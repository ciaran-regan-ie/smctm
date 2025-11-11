from omegaconf import DictConfig

from tasks.task import Task
from tasks.cifarfewshot import CIFARFewShotTask

__all__ = ["CIFARFewShot"]

_TASKS = {"CIFARFewShot": CIFARFewShotTask}

# Constructs a task based on the config
def construct_task(cfg: DictConfig, logger=None) -> Task:
	return _TASKS[cfg.task.type](cfg, logger=logger)