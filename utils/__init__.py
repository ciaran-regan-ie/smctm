import random

import numpy as np
import torch

from utils.logging import LocalLogger
from utils.plotting import plot_metrics

__all__ = ["LocalLogger", "plot_metrics", "set_random_seeds"]


def set_random_seeds(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)  # Sets seed for all devices