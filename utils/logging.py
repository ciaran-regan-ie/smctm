import json
import os
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None


class LocalLogger:
	def __init__(self, cfg: dict):
		self.metrics = {}  # Stores metrics as {'name': [(step_a, value_a), (step_b, value_b), ...]}
		
		wandb_cfg = cfg.get("wandb", {})
		self.use_wandb = wandb_cfg.get("enabled", False) and wandb is not None
		
		if self.use_wandb:
			# Create run name from task, model, and timestamp
			task_type = cfg.get("task", {}).get("type", "")
			model_type = cfg.get("model", {}).get("type", "")
			timestamp = os.path.basename(os.getcwd())
			task_subtype = cfg.get("task", {}).get("subtype")
			if task_subtype is not None:
				task_type += f"_{task_subtype}"
			run_name = f"{task_type}_{model_type}_{timestamp}"
			
			wandb.init(
				project=wandb_cfg.get("project", "smctm"),
				entity=wandb_cfg.get("entity", None),
				config=cfg,
				tags=wandb_cfg.get("tags", []),
				name=run_name,
			)

	def log(self, name: str, value: float, step: int):
		# Local logging
		if name not in self.metrics:
			self.metrics[name] = []
		self.metrics[name].append((step, value))

		# Weights & Biases logging
		if self.use_wandb:
			wandb.log({name: value}, step=step)

	def get_metrics(self) -> dict:
		return self.metrics

	def save(self, filepath: str):
		with open(filepath, "w") as f:
			json.dump(self.metrics, f)
	
	def finish(self):
		"""Finish wandb run if active"""
		if self.use_wandb:
			wandb.finish()