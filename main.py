import json
import logging
import timeit

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tasks import construct_task
from utils import LocalLogger, plot_metrics, set_random_seeds

log = logging.getLogger(__name__)

def log_metrics(logger, metrics: dict[str, float], step: int, prefix: str = ""):
	for name, value in metrics.items():
		logger.log(f"{prefix}_{name}", value, step)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
	log.info(OmegaConf.to_yaml(cfg))
	set_random_seeds(cfg.seed)
	
	logger = LocalLogger(cfg=OmegaConf.to_container(cfg, resolve=True))
	summary = {}

	task = construct_task(cfg, logger=logger)

	timer = timeit.default_timer()
	pbar = tqdm(range(1, cfg.epochs + 1), desc="Training")
	for epoch in pbar:
		
		# Train and evaluate
		train_metrics = task.train(epoch)
		log_metrics(logger, train_metrics, task.global_step, prefix="train")

		# Evaluate
		if epoch % cfg.eval_interval == 0:
			eval_metrics = task.eval(epoch)
			log_metrics(logger, eval_metrics, task.global_step, prefix="eval")

		# Plot
		if epoch % cfg.plot_interval == 0:
			plot_metrics(logger.get_metrics())
		pbar.set_description(str(eval_metrics))
	logger.save("metrics.json")

	# Calculate run summary
	summary["parameter_count"] = task.get_parameter_count()
	summary["runtime"] = timeit.default_timer() - timer
	summary["performance"] = task.calculate_performance(logger.get_metrics())
	with open("summary.json", "w") as f:
		json.dump(summary, f)

	task.perform_diagnostics()  # Perform final task/model diagnostics
	log.info(f"Performance: {summary['performance']}")
	
	# Finish wandb run
	logger.finish()
	
	return summary["performance"]


if __name__ == "__main__":
	main()