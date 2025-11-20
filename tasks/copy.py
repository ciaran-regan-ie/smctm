import torch
import torch.nn as nn
from torch.utils.data import Dataset
from omegaconf import DictConfig
from torch.optim import AdamW
from tqdm import tqdm
from omegaconf import OmegaConf

from models import construct_model
from tasks.task import Task
from tasks.utils import get_device

class CopyTask(Task):
    def __init__(self, cfg: DictConfig, logger=None):
        super().__init__()
        self.device = get_device(cfg)
        self.logger = logger
        self.global_step = 0
        self.copy_dim = cfg.task.copy_dim
        self.copy_sequence_length = cfg.task.copy_sequence_length
        self.copy_delay = cfg.task.copy_delay

        train_data = CopyDataset(dim=cfg.task.copy_dim, seq_length=cfg.task.copy_sequence_length, delay=cfg.task.copy_delay)
        test_data = CopyDataset(dim=cfg.task.copy_dim, seq_length=cfg.task.copy_sequence_length, delay=cfg.task.copy_delay)

        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=1, drop_last=False)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True, num_workers=1, drop_last=False)

        assert cfg.task.out_dims == cfg.task.copy_dim
        self.model = construct_model(model_cfg=cfg.model, task_cfg=cfg.task).to(self.device)
        self.loss = MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        assert cfg.learn_rate_scheduler in ("none", "cosine_annealing", "linear", "cosine_annealing_warm_restarts", "multi_step")
        if cfg.learn_rate_scheduler == "none":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        elif cfg.learn_rate_scheduler == "cosine_annealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * cfg.learn_rate_scheduler_final_factor)
        elif cfg.learn_rate_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=cfg.learn_rate_scheduler_final_factor, total_iters=cfg.epochs)
        elif cfg.learn_rate_scheduler == "cosine_annealing_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=cfg.learning_rate * cfg.learn_rate_scheduler_final_factor)
        elif cfg.learn_rate_scheduler == "multi_step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(cfg.epochs*0.5), int(cfg.epochs*0.75)], gamma=0.1)
        
        self.gradient_clipping = cfg.gradient_clipping
        self.init_lazy_modules()
        print(f"Total Parameter Count: {self.get_parameter_count()}")
        print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            logits = logits.transpose(1, 2)  # (batch, dim, time) -> (batch, time, dim)
            loss = self.loss(logits, targets)
            loss.backward()
    
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimizer.step()
            total_loss += loss.item()
                        
            if self.logger:
                self.logger.log("train_loss_batch", loss.item(), self.global_step)
                self.global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        self.scheduler.step()
        
        return {"loss": total_loss / len(self.train_dataloader), "lr": self.scheduler.get_last_lr()[0]}

    def calculate_accuracy(self):
        """Copy task does not use accuracy."""
        pass

    def eval(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0
        for inputs, targets in tqdm(self.test_dataloader, leave=False, desc=f"Eval Epoch {epoch}"):
            with torch.inference_mode():
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs)
                predictions = predictions.transpose(1, 2)  # (batch, dim, time) -> (batch, time, dim)
                loss = self.loss(predictions, targets)
                total_loss += loss.item()
        return {"loss": total_loss / len(self.test_dataloader)}

    def calculate_performance(self, metrics: dict[str, float], window_size: int = 5) -> float:
        accuracies = [pair[1] for pair in metrics["eval_loss"]]
        recent_accuracies = accuracies[-window_size:]
        return sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0.0

    def init_lazy_modules(self):
        B = 1
        input_shape = (B, (2 * self.copy_sequence_length) + self.copy_delay, self.copy_dim + 2) # +2 for flags
        pseudo_inputs = torch.zeros(input_shape, device=self.device).float()
        self.model(pseudo_inputs)
        

# --- Dataset Logic ---

class CopyDataset(Dataset):

    def __init__(self, dim: int, seq_length: int, delay: int):

        self.dim = dim
        self.seq_length = seq_length
        self.delay = delay

    def __len__(self):
        return 10000

    def __getitem__(self, index):

        seq = torch.randn((self.seq_length, self.dim))
        sample_input = torch.cat((
            seq,
            torch.ones((self.seq_length, 1)),
            torch.zeros((self.seq_length, 1)),
        ), dim=1)

        delay_input = torch.zeros((self.delay, self.dim + 2))
        test_input = torch.zeros((self.seq_length, self.dim + 2))
        test_input[:, -1] = 1

        input = torch.cat((sample_input, delay_input, test_input), dim=0)
        output = torch.cat((torch.zeros((self.seq_length + self.delay, self.dim)), seq), dim=0)

        return input, output

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)
