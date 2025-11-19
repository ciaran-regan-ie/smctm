import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm

from models import construct_model
from tasks.task import Task
from tasks.utils import get_device

class FewShotImageClassificationTask(Task):
    def __init__(self, cfg: DictConfig, logger=None):
        super().__init__()
        self.device = get_device(cfg)
        self.logger = logger
        self.global_step = 0
        
        train_base_dataset = FewShotDataset(dataset_name=cfg.task.dataset, phase='train')
        test_base_dataset = FewShotDataset(dataset_name=cfg.task.dataset, phase='test')
        
        self.train_dataloader = FewShotDataloader(
            dataset=train_base_dataset,
            internal_ticks=cfg.task.iterations,
            nKnovel=cfg.task.nKnovel,
            nKbase=cfg.task.nKbase,
            nExemplars=cfg.task.nExemplars,
            nTestNovel=cfg.task.nTestNovel,
            nTestBase=cfg.task.nTestBase,
            epoch_size=cfg.batch_size * 1000,
            batch_size=cfg.batch_size
        )
        self.test_dataloader = FewShotDataloader(
            dataset=test_base_dataset,
            internal_ticks=cfg.task.iterations,
            nKnovel=cfg.task.nKnovel,
            nKbase=cfg.task.nKbase,
            nExemplars=cfg.task.nExemplars,
            nTestNovel=cfg.task.nTestNovel,
            nTestBase=cfg.task.nTestBase,
            epoch_size=cfg.batch_size * 25,
            batch_size=cfg.batch_size
        )
        self.model = construct_model(model_cfg=cfg.model, task_cfg=cfg.task).to(self.device)
        self.loss = FewShotLoss(
            iterations_per_image=cfg.task.iterations, 
            num_test_images=cfg.task.nTestNovel+cfg.task.nTestBase, 
            num_classes=cfg.task.out_dims
        )
        self.optimiser = AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        assert cfg.learn_rate_scheduler in ("none", "cosine_annealing", "linear", "cosine_annealing_warm_restarts", "multi_step")
        if cfg.learn_rate_scheduler == "none":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lambda epoch: 1.0)
        elif cfg.learn_rate_scheduler == "cosine_annealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=cfg.epochs, eta_min=cfg.learning_rate * cfg.learn_rate_scheduler_final_factor)
        elif cfg.learn_rate_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimiser, start_factor=1, end_factor=cfg.learn_rate_scheduler_final_factor, total_iters=cfg.epochs)
        elif cfg.learn_rate_scheduler == "cosine_annealing_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=10, T_mult=1, eta_min=cfg.learning_rate * cfg.learn_rate_scheduler_final_factor)
        elif cfg.learn_rate_scheduler == "multi_step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=[int(cfg.epochs*0.5), int(cfg.epochs*0.75)], gamma=0.1)
        
        self.gradient_clipping = cfg.gradient_clipping
        self.init_lazy_modules()
        print(f"Total Parameter Count: {self.get_parameter_count()}")
        print(f"Backbone Parameter Count: {self.get_backbone_parameter_count()}")

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for (inputs, aux_inputs), targets in pbar:
            inputs, aux_inputs, targets = inputs.to(self.device), aux_inputs.to(self.device), targets.to(self.device)
            self.optimiser.zero_grad()
            logits = self.model(inputs, aux_inputs)
            loss, info = self.loss(logits, targets)
            loss.backward()
    
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimiser.step()
            total_loss += loss.item()
            
            train_accuracy_batch = self.calculate_accuracy(logits, info, targets)
            train_accuracy_all_images_batch = self.calculate_accuracy_all_images(logits, info, targets)
            
            if self.logger:
                self.logger.log("train_loss_batch", loss.item(), self.global_step)
                self.logger.log("train_accuracy_batch", train_accuracy_batch, self.global_step)
                self.logger.log("train_accuracy_all_images_batch", train_accuracy_all_images_batch, self.global_step)
                self.global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}","accuracy": f"{train_accuracy_batch:.4f}", "accuracy_all_images": f"{train_accuracy_all_images_batch:.4f}"})
        self.scheduler.step()
        
        return {"loss": total_loss / len(self.train_dataloader), "lr": self.scheduler.get_last_lr()[0]}

    def calculate_accuracy(self, predictions, info, targets):
        answer_timestep = info["prediction_timesteps"][0]
        targets = targets[:, answer_timestep]
        predictions = predictions[:, :, answer_timestep].argmax(1)
        accuracy = (predictions == targets).float().mean().item()
        return accuracy

    def calculate_accuracy_all_images(self, predictions, info, targets):
        answer_timesteps = info["prediction_timesteps"]
        timesteps_tensor = torch.tensor(answer_timesteps, device=predictions.device)
        targets_at_timesteps = targets[:, timesteps_tensor]
        predictions_at_timesteps = predictions[:, :, timesteps_tensor].argmax(dim=1)
        accuracy = (predictions_at_timesteps == targets_at_timesteps).float().mean().item()
        return accuracy

    def eval(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss, accuracy, accuracy_all_images = 0, 0, 0
        for (inputs, aux_inputs), targets in tqdm(self.test_dataloader, leave=False, desc=f"Eval Epoch {epoch}"):
            with torch.inference_mode():
                inputs, aux_inputs, targets = inputs.to(self.device), aux_inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs, aux_inputs)
                loss, info = self.loss(predictions, targets)
                total_loss += loss.item()
                accuracy += self.calculate_accuracy(predictions, info, targets)
                accuracy_all_images += self.calculate_accuracy_all_images(predictions, info, targets)
        return {"loss": total_loss / len(self.test_dataloader), "accuracy": accuracy /  len(self.test_dataloader), "accuracy_all_images": accuracy_all_images /  len(self.test_dataloader)}

    def calculate_performance(self, metrics: dict[str, float], window_size: int = 5) -> float:
        accuracies = [pair[1] for pair in metrics["eval_accuracy"]]
        recent_accuracies = accuracies[-window_size:]
        return sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0.0

    def init_lazy_modules(self):
        B, T = 1, self.train_dataloader.internal_ticks
        total_internal_ticks = T * (self.train_dataloader.nKnovel+self.train_dataloader.nKbase+self.train_dataloader.nTestNovel+self.train_dataloader.nTestBase)
        
        # Get image size from dataset config
        img_size = self.train_dataloader.dataset.config['image_size']
        input_shape = (B, total_internal_ticks, 3, img_size, img_size)
        aux_input_shape = (B, total_internal_ticks, self.train_dataloader.nKnovel)
        
        aux_inputs = torch.zeros(aux_input_shape, device=self.device).float()
        pseudo_inputs = torch.zeros(input_shape, device=self.device).float()
        self.model(pseudo_inputs, aux_inputs=aux_inputs)
        
class FewShotLoss(nn.Module):
    def __init__(self, iterations_per_image, num_test_images, num_classes):
        super().__init__()
        self.iterations_per_image = iterations_per_image
        self.num_test_images = num_test_images
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        predictions = predictions[:, :, -(self.iterations_per_image * self.num_test_images):]
        predictions = predictions.transpose(-2, -1)
        predictions = predictions.reshape(-1, self.num_classes)

        targets = targets[:, -(self.iterations_per_image * self.num_test_images):]
        targets = targets.reshape(-1)

        loss = nn.CrossEntropyLoss()(predictions, targets)

        info = {
            'prediction_timesteps': [((self.iterations_per_image * self.num_test_images) + (self.iterations_per_image * i + (self.iterations_per_image - 1))) for i in range(self.num_test_images)]
        }

        return loss, info


# --- Dataset Logic ---

import os
import numpy as np
import random
import pickle
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchnet as tnt
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset configurations
DATASET_CONFIGS = {
    'CIFAR100': {
        'dir': os.path.join(_PROJECT_ROOT, 'data', 'cifar_fs'),
        'mean': [129.37731888, 124.10583864, 112.47758569],
        'std': [68.20947949, 65.43124043, 70.45866994],
        'image_size': 32,
        'padding': 4,
        'train_file': 'CIFAR_FS_train.pickle',
        'val_file': 'CIFAR_FS_val.pickle',
        'test_file': 'CIFAR_FS_test.pickle',
    },
    'MiniImageNet': {
        'dir': os.path.join(_PROJECT_ROOT, 'data', 'miniImageNet'),
        'mean': [120.39586422, 115.59361427, 104.54012653],
        'std': [70.68188272, 68.27635443, 72.54505529],
        'image_size': 84,
        'padding': 8,
        'train_file': 'miniImageNet_category_split_train_phase_train.pickle',
        'val_file': 'miniImageNet_category_split_val.pickle',
        'test_file': 'miniImageNet_category_split_test.pickle',
        'train_val_file': 'miniImageNet_category_split_train_phase_val.pickle',
        'train_test_file': 'miniImageNet_category_split_train_phase_test.pickle',
    }
}

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds

def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

class FewShotDataset(data.Dataset):
    def __init__(self, dataset_name, phase='train', do_not_use_random_transf=False):
        assert dataset_name in DATASET_CONFIGS, f"Dataset {dataset_name} not supported"
        assert phase in ['train', 'val', 'test'], f"Phase {phase} not valid"
        
        self.dataset_name = dataset_name
        self.phase = phase
        self.name = f'{dataset_name}_{phase}'
        self.config = DATASET_CONFIGS[dataset_name]
        
        print(f'Loading {dataset_name} dataset - phase {phase}')
        
        if phase == 'train':
            data_train = load_data(os.path.join(self.config['dir'], self.config['train_file']))
            self.data = data_train['data']
            self.labels = data_train['labels']
            
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
            
        elif phase in ['val', 'test']:
            if dataset_name == 'CIFAR100':
                data_base = load_data(os.path.join(self.config['dir'], self.config['train_file']))
                novel_file = self.config['test_file'] if phase == 'test' else self.config['val_file']
                data_novel = load_data(os.path.join(self.config['dir'], novel_file))
            else:  # miniImageNet
                base_file = self.config['train_test_file'] if phase == 'test' else self.config['train_val_file']
                data_base = load_data(os.path.join(self.config['dir'], base_file))
                novel_file = self.config['test_file'] if phase == 'test' else self.config['val_file']
                data_novel = load_data(os.path.join(self.config['dir'], novel_file))
            
            self.data = np.concatenate([data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']
            
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            
            self.labelIds_base = list(buildLabelIndex(data_base['labels']).keys())
            self.labelIds_novel = list(buildLabelIndex(data_novel['labels']).keys())
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0, "Base and novel categories should not overlap"
        
        mean_pix = [x/255.0 for x in self.config['mean']]
        std_pix = [x/255.0 for x in self.config['std']]
        self.normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        
        class ToTensor(torchvision.transforms.ToTensor):
            def __call__(self, pic):
                return super().__call__(np.array(pic, copy=True))
        
        if (phase in ['test', 'val']) or do_not_use_random_transf:
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                ToTensor(),
                self.normalize
            ])
        else:
            img_size = self.config['image_size']
            padding = self.config['padding']
            self.transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=padding),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                ToTensor(),
                self.normalize
            ])
    
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 internal_ticks=10,
                 nKnovel=5,
                 nKbase=-1,
                 nExemplars=1,
                 nTestNovel=15*5,
                 nTestBase=15*5,
                 batch_size=1,
                 num_workers=8,
                 epoch_size=2000):

        self.dataset = dataset
        self.phase = self.dataset.phase
        self.internal_ticks = internal_ticks

        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)
        assert(nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            Kbase = sorted(self.sampleCategories('base', nKbase))
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        Tbase = []
        if len(Kbase) > 0:
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert(len(Tbase) == nTestBase)
        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for Knovel_idx in range(len(Knovel)):
            imd_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_ememplars = imd_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, nKbase+Knovel_idx) for img_id in imds_tnovel]
            Exemplars += [(img_id, nKbase+Knovel_idx) for img_id in imds_ememplars]
        assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples, num_classes=None, is_support=False):
        if len(examples) == 0:
            # Get image size from dataset config
            img_size = self.dataset.config['image_size']
            return torch.empty(0, 3, img_size, img_size), torch.empty(0, 0), torch.empty(0, dtype=torch.long)

        images_list, onehots_list, labels_list = [], [], []

        if num_classes is None:
            num_classes = max(label for _, label in examples) + 1

        # Get image size from dataset config
        img_size = self.dataset.config['image_size']

        for img_idx, label in examples:
            img, _ = self.dataset[img_idx]  # [C, H, W]
            repeated_img = img.unsqueeze(0).repeat(self.internal_ticks, 1, 1, 1)

            if is_support:
                onehot = torch.zeros(self.internal_ticks, num_classes, dtype=torch.float32)
                onehot[:, label] = 1.0
            else:
                onehot = torch.zeros(self.internal_ticks, num_classes, dtype=torch.float32)

            repeated_label = torch.full((self.internal_ticks,), label, dtype=torch.long)

            images_list.append(repeated_img)
            onehots_list.append(onehot)
            labels_list.append(repeated_label)

        images = torch.cat(images_list, dim=0)
        onehots = torch.cat(onehots_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        return images, onehots, labels

    def get_iterator(self, epoch=0):
        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            num_classes = len(Kall)

            Xt, Yt_onehot, Yt_labels = self.createExamplesTensorData(Test, num_classes=num_classes, is_support=False)

            if len(Exemplars) > 0:
                Xe, Ye_onehot, Ye_labels = self.createExamplesTensorData(Exemplars, num_classes=num_classes, is_support=True)
                inputs = torch.cat([Xe, Xt], dim=0)
                aux_inputs = torch.cat([Ye_onehot, Yt_onehot], dim=0)
                targets = torch.cat([Ye_labels, Yt_labels], dim=0)
            else:
                inputs, aux_inputs, targets = Xt, Yt_onehot, Yt_labels

            return (inputs, aux_inputs), targets

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __iter__(self):
        return iter(self.get_iterator())

    def __len__(self):
        return int(self.epoch_size / self.batch_size)
    