from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset

from common.common import parse_args
import models.classifier as C
from datasets.datasets import get_dataset_2, get_style_dataset, BalancedMultiSourceRandomSampler
from utils.utils import load_ckpt
import sys
import os
from torchlars import LARS

P = parse_args()

# default settings
P.batch_p = 2
P.adapt_batch_K = True
P.adapt_lr = True
P.replay_selection_strategy = "random"
P.new_classes_mapping = "random"
P.model = "resnet18"
P.im_mean = [0.485, 0.456, 0.406]
P.im_std = [0.229, 0.224, 0.225]
P.batch_size = 32
P.eval_alpha_multiplier = 1.0

if P.dataset == "COSDA-HR":
    P.source = "source"
    P.total_episodes = 16
    P.batch_K = 10
elif P.dataset == "OWR":
    P.total_episodes = 4
    P.batch_K = 11
elif P.dataset == "CORe50":
    P.source = "source"
    P.total_episodes = 4
    P.batch_K = 10

# an estimate of total number of iterations which however could be much more
P.iterations = P.ep_0_min_its + (P.total_episodes - 1) * P.eps_min_its

### Set torch device ###
assert torch.cuda.is_available(), "We support only training on CUDA!"
device = torch.device("cuda")

P.n_gpus = 1

from tqdm import tqdm
import numpy as np

P.local_rank = 0
P.n_gpus = 1
print(f"Num of GPUS {P.n_gpus}")

P.current_episode = 0
### Initialize dataset ###
train_sets = get_dataset_2(P, episode=P.current_episode)

# we have a list of ConcatDatasets, one for each class
P.image_size = (224, 224, 3)
P.n_classes = P.total_n_classes
kwargs = {'pin_memory': False, 'num_workers': P.num_workers, 'drop_last':True}

assert P.batch_K%P.n_gpus == 0, "batch_K has to be divisible by world size!!"
single_GPU_batch_K = P.batch_K/P.n_gpus
single_GPU_batch_size = int(P.batch_p*single_GPU_batch_K)
whole_source = ConcatDataset(train_sets)
my_sampler = BalancedMultiSourceRandomSampler(whole_source, P.batch_p, P.local_rank, P.n_gpus)
print(f"Rank {P.local_rank}: sampler_size: {len(my_sampler)}. Dataset_size: {len(whole_source)}")
train_loader = DataLoader(whole_source, sampler=my_sampler, batch_size=single_GPU_batch_size, **kwargs)

# test dataset -> returns only current episode train sample with eval transforms
source_datasets_test = get_dataset_2(P, episode=P.current_episode, train=False)
whole_source_test = ConcatDataset(source_datasets_test)
source_test_loader = DataLoader(whole_source_test, shuffle=False, batch_size=1, num_workers=P.num_workers)

### Initialize model ###
# define transformations for SimCLR augmentation
simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)

# generate model (includes backbone + classification head)
model = C.get_classifier(P.model, n_classes=P.n_classes, simclr_dim=P.simclr_dim).to(device)

# modify normalize params if necessary
mean = torch.tensor(P.im_mean).to(device)
std = torch.tensor(P.im_std).to(device)
model.normalize.mean=mean
model.normalize.std=std

criterion = nn.CrossEntropyLoss().to(device)

# wrap SGD in LARS for multi-gpu optimization
base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
lr_decay_gamma = 0.1

# our scheduler is warmup + constant (obtained via step lr without steps)
#scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=[P.iterations])

# a warmup scheduler is used in the first iterations, then substituted with the scheduler defined above
from training.scheduler import GradualWarmupScheduler, ConstantScheduler
scheduler = ConstantScheduler(optimizer)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=P.warmup, after_scheduler=scheduler)
