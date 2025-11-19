from omegaconf import DictConfig

from tasks.task import Task
from tasks.few_shot_cifar import FewShotCIFARTask
from tasks.few_shot_miniimagenet import FewShotMiniImageNetTask

__all__ = ["FewShotCIFAR", "FewShotMiniImageNet"]

_TASKS = {
    "FewShotCIFAR": FewShotCIFARTask,
    "FewShotMiniImageNet": FewShotMiniImageNetTask,
}

# Constructs a task based on the config
def construct_task(cfg: DictConfig, logger=None) -> Task:
	return _TASKS[cfg.task.type](cfg, logger=logger)