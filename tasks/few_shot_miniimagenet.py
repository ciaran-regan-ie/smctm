from omegaconf import DictConfig, OmegaConf
from tasks.few_shot_image_classification import (
    FewShotImageClassificationTask,
    FewShotDataset,
    FewShotDataloader,
)


class FewShotMiniImageNetTask(FewShotImageClassificationTask):
    """MiniImageNet Few-Shot Learning Task.
    
    Extends the base FewShotImageClassificationTask with MiniImageNet specific configuration.
    All dataloading and training logic is inherited from the base class.
    """
    
    def __init__(self, cfg: DictConfig, logger=None):
        # Set the dataset to MiniImageNet
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        cfg_copy['task']['dataset'] = 'MiniImageNet'
        cfg = OmegaConf.create(cfg_copy)
        
        super().__init__(cfg, logger)