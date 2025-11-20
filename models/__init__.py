from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig
from torch import nn

from models.ctm import ContinuousThoughtMachine as CTM
from models.lstm import LSTM
from models.backbones import ProtoNetEmbedding
from models.interactions import FeedForwardDataInteraction, FewShotImageClassificationDataInteraction

_MODELS = {"CTM": CTM, "LSTM": LSTM}

def construct_backbone(model_cfg: DictConfig, task_cfg: DictConfig):
	if task_cfg.type in ("FewShotImageClassification", "FewShotCIFAR", "FewShotMiniImageNet"):
		return ProtoNetEmbedding(3, 64, 64)
	elif task_cfg.type in ("Copy",):
		return nn.Identity()
	else:
		raise ValueError(f"Unsupported task type: {task_cfg.type}")

def construct_data_interaction(model_cfg: DictConfig, task_cfg: DictConfig):
	if task_cfg.type in ("FewShotImageClassification", "FewShotCIFAR", "FewShotMiniImageNet"):
		return FewShotImageClassificationDataInteraction(backbone=construct_backbone(model_cfg, task_cfg), d_input=task_cfg.d_input, use_output_proj=True)
	elif task_cfg.type in ("Copy",):
		d_input = task_cfg.copy_dim + 2  # +2 for flags
		return FeedForwardDataInteraction(backbone=construct_backbone(model_cfg, task_cfg), d_input=d_input, use_output_proj=False)
	else:
		raise ValueError(f"Unsupported task type: {task_cfg.type}")

# Constructs model based the model config
def construct_model(model_cfg: DictConfig, task_cfg: DictConfig) -> nn.Module:
	model_type = model_cfg.type
	del model_cfg.type  # Remove type before being used to instantiate the model
	return _MODELS[model_type](data_interaction=construct_data_interaction(model_cfg, task_cfg), out_dims=task_cfg.out_dims, **model_cfg)
