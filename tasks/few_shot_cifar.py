from omegaconf import DictConfig, OmegaConf
from tasks.few_shot_image_classification import (
    FewShotImageClassificationTask,
    FewShotDataset,
    FewShotDataloader,
)


class FewShotCIFARTask(FewShotImageClassificationTask):
    """CIFAR-100 Few-Shot Learning Task.
    
    Extends the base FewShotImageClassificationTask with CIFAR-100 specific configuration.
    All dataloading and training logic is inherited from the base class.
    """
    
    def __init__(self, cfg: DictConfig, logger=None):
        # Set the dataset to CIFAR100
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        cfg_copy['task']['dataset'] = 'CIFAR100'
        cfg = OmegaConf.create(cfg_copy)
        
        super().__init__(cfg, logger)
