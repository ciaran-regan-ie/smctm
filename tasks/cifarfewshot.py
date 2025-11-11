import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm
from torch.optim import AdamW

from models import construct_model
from tasks.task import Task
from tasks.utils import get_device

class CIFARFewShotTask(Task):
    def __init__(self, cfg: DictConfig, logger=None):
        super().__init__()
        self.device = get_device(cfg)
        self.logger = logger
        self.global_step = 0  # Track global step for batch-level logging
        train_base_dataset = CIFARFewShotDataset(phase='train')
        test_base_dataset = CIFARFewShotDataset(phase='test')
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
        self.loss = CIFARFewShotLoss(iterations_per_image=cfg.task.iterations, num_test_images=cfg.task.nTestNovel+cfg.task.nTestBase, num_classes=cfg.task.out_dims)
        self.optimiser = AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.gradient_clipping = cfg.gradient_clipping
        self.init_lazy_modules()

    def train(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_dataloader, leave=False, desc=f"Epoch {epoch}")
        for (inputs, aux_inputs), targets in pbar:
            inputs, aux_inputs, targets = inputs.to(self.device), aux_inputs.to(self.device), targets.to(self.device)
            self.optimiser.zero_grad()
            logits = self.model(inputs, aux_inputs)
            loss, _ = self.loss(logits, targets)
            loss.backward()
    
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimiser.step()
            total_loss += loss.item()
            
            if self.logger:
                self.logger.log("train_loss_batch", loss.item(), self.global_step)
                self.global_step += 1
            
            # Update progress bar with current batch loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return {"loss": total_loss / len(self.train_dataloader)}

    def eval(self, epoch: int) -> dict[str, float]:

        def calculate_accuracy(predictions, info, targets):
            # For CifarFewShot we calculate the accuracy as the accuracy of the first test image only.
            answer_timestep = info["loss_index_2"]
            targets = targets[:, answer_timestep]
            predictions = predictions.reshape(predictions.size(0), -1, 5, predictions.size(-1)) # Assuming 5 classes
            pred_at_answer_timestep = predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :, answer_timestep]
            correct = (pred_at_answer_timestep == targets).sum().item()
            return correct

        self.model.eval()
        total_loss, accuracy = 0, 0
        for (inputs, aux_inputs), targets in tqdm(self.test_dataloader, leave=False, desc=f"Eval Epoch {epoch}"):
            with torch.inference_mode():
                inputs, aux_inputs, targets = inputs.to(self.device), aux_inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs, aux_inputs)
                loss, info = self.loss(predictions, targets)
                total_loss += loss.item()
                accuracy += calculate_accuracy(predictions, info, targets)

        return {"loss": total_loss / len(self.test_dataloader), "accuracy": accuracy /  (self.test_dataloader.epoch_size*self.test_dataloader.batch_size)}

    def calculate_performance(self, metrics: dict[str, float]) -> float:
        accuracies = [pair[1] for pair in metrics["eval_accuracy"]]
        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    def init_lazy_modules(self):
        B, T = 1, self.train_dataloader.internal_ticks
        total_internal_ticks = T * (self.train_dataloader.nKnovel+self.train_dataloader.nKbase+self.train_dataloader.nTestNovel+self.train_dataloader.nTestBase)
        input_shape = (B, total_internal_ticks, 3, 32, 32)
        aux_input_shape = (B, total_internal_ticks, self.train_dataloader.nKnovel)
        aux_inputs = torch.zeros(aux_input_shape, device=self.device).float()
        pseudo_inputs = torch.zeros(input_shape, device=self.device).float()
        self.model(pseudo_inputs, aux_inputs=aux_inputs, track=False)
        pass
        
class CIFARFewShotLoss(nn.Module):
    def __init__(self, iterations_per_image, num_test_images, num_classes):
        super().__init__()

        self.iterations_per_image = iterations_per_image
        self.num_test_images = num_test_images
        self.num_classes = num_classes

    def forward(self, predictions, targets):

        predictions = predictions[:, :, -(self.iterations_per_image * self.num_test_images):]  # (B, num_classes, iterations_per_image * num_test_images)
        predictions = predictions.transpose(-2, -1)  # (B, iterations_per_image * num_test_images, num_classes)
        predictions = predictions.reshape(-1, self.num_classes)  # (B * iterations_per_image * num_test_images, num_classes)

        targets = targets[:, -(self.iterations_per_image * self.num_test_images):]  # (B, iterations_per_image * num_test_images)
        targets = targets.reshape(-1)  # (B * iterations_per_image * num_test_images,)

        # Compute cross-entropy loss
        loss = nn.CrossEntropyLoss()(predictions, targets)

        # Following the evaluation procedure of https://arxiv.org/abs/2302.03235, we only consider the first image when calculating the accuracy
        # For the CTM, which process each image for self.iterations_per_image steps, we take the last thought step of the first image
        info = {
            'loss_index_2': (self.iterations_per_image * self.num_test_images) + (self.iterations_per_image - 1)
        }

        return loss, info




# --- Dataset Logic ---

# Dataloader of Gidaris & Komodakis, CVPR 2018
# Modified to repeat images across time steps for temporal processing

# from __future__ import print_function
import os
import os.path
import numpy as np
import random
import pickle
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchnet as tnt
from PIL import Image

# Set the appropriate paths of the datasets here.
# Use absolute path to handle Hydra changing working directory
# Get the project root directory (parent of tasks/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CIFAR_FS_DATASET_DIR = os.path.join(_PROJECT_ROOT, 'data', 'cifar_fs')

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

class CIFARFewShotDataset(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False):

        assert(phase=='train' or phase=='val' or phase=='test')
        self.phase = phase
        self.name = 'CIFAR_FS_' + phase

        file_train_categories_train_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_train_categories_val_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_train_categories_test_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_val_categories_val_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_test.pickle')

        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_test_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_test_categories_test_phase)
            else: # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_val_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_val_categories_val_phase)

            self.data = np.concatenate(
                [data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            # convert dict_keys view to list to avoid sampling from a set-like
            # object (random.sample deprecates sampling from sets/dict_keys)
            self.labelIds_base = list(buildLabelIndex(data_base['labels']).keys())
            self.labelIds_novel = list(buildLabelIndex(data_novel['labels']).keys())
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        mean_pix = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        
        self.normalize = normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        class ToTensor(torchvision.transforms.ToTensor):
            def __call__(self, pic):
                return super().__call__(np.array(pic, copy=True))

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                ToTensor(),
                normalize
            ])
        else:
            
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                ToTensor(),
                normalize
            ])
            
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 internal_ticks=10,  # NEW: Number of time steps to repeat each image
                 nKnovel=5, # number of novel categories.
                 nKbase=-1, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 nTestBase=15*5, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=8,
                 epoch_size=2000, # number of batches per epoch.
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        self.internal_ticks = internal_ticks  # NEW: Store internal_ticks

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
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).
        """
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.
        """
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.
        """
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
        """Sample `nTestBase` number of images from the `Kbase` categories."""
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
        """Samples train and test examples of the novel categories."""

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
        """Samples a training episode."""

        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples, num_classes=None, is_support=False):
        if len(examples) == 0:
            return torch.empty(0, 3, 32, 32), torch.empty(0, 0), torch.empty(0, dtype=torch.long)

        images_list, onehots_list, labels_list = [], [], []

        if num_classes is None:
            num_classes = max(label for _, label in examples) + 1

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

        images = torch.cat(images_list, dim=0)    # [T, C, H, W]
        onehots = torch.cat(onehots_list, dim=0)  # [T, num_classes]
        labels = torch.cat(labels_list, dim=0)    # [T]

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
