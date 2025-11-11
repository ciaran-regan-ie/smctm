from abc import ABC, abstractmethod

class Task(ABC):
	@abstractmethod
	def train(self, epoch: int) -> dict[str, float]:
		pass

	@abstractmethod
	def eval(self, epoch: int) -> dict[str, float]:
		pass

	@abstractmethod
	def calculate_performance(self, metrics: dict[str, float]) -> float:
		pass

	@abstractmethod
	def init_lazy_modules(self):
		pass

	def get_parameter_count(self) -> int:
		return sum(param.numel() for param in self.model.parameters())

	def perform_diagnostics(self):
		pass
