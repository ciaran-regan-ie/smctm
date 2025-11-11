from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig
from torch import nn

from models.ctm import ContinuousThoughtMachine as CTM
from models.backbones import ProtoNetEmbedding
from models.interactions import CIFARFSDataInteraction

_MODELS = {"CTM": CTM}

def construct_backbone(model_cfg: DictConfig, task_cfg: DictConfig):
	if task_cfg.type == "CIFARFewShot":
		return ProtoNetEmbedding(3, 64, 64)
	else:
		raise ValueError(f"Unsupported task type: {task_cfg.task}")

def construct_data_interaction(model_cfg: DictConfig, task_cfg: DictConfig):
	if task_cfg.type == "CIFARFewShot":
		return CIFARFSDataInteraction(backbone=construct_backbone(model_cfg, task_cfg), d_input=model_cfg.d_input)
	else:
		raise ValueError(f"Unsupported task type: {task_cfg.task}")

# Constructs model based the model config
def construct_model(model_cfg: DictConfig, task_cfg: DictConfig) -> nn.Module:
	model_type = model_cfg.type
	del model_cfg.type  # Remove type before being used to instantiate the model
	return _MODELS[model_type](data_interaction=construct_data_interaction(model_cfg, task_cfg), out_dims=task_cfg.out_dims, **model_cfg)
