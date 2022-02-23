from copy import deepcopy

import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from common.common import parse_args
import models.classifier as C
from datasets.datasets import get_dataset_2, get_test_datasets, get_replay_datasets
from utils.utils import load_ckpt

P = parse_args()

### Set torch device ###
P = parse_args()
assert torch.cuda.is_available(), "We support only training on CUDA!"
device = torch.device("cuda")

P.n_gpus = 1
P.multi_gpu = False
P.local_rank = 0
P.image_size = (224, 224, 3)
P.n_classes = P.total_n_classes
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
    P.target = "target"
    P.total_episodes = 16
    P.batch_K = 10


### Initialize model ###
model = C.get_classifier(P.model, n_classes=P.n_classes,simclr_dim=P.simclr_dim).to(device)

criterion = nn.CrossEntropyLoss().to(device)

# set normalize params
mean = torch.tensor(P.im_mean).to(device)
std = torch.tensor(P.im_std).to(device)
model.normalize.mean=mean
model.normalize.std=std

# load model
assert P.load_path is not None, "You need to pass checkpoint path using --load_path"
model_state_dict, aux_data, _ = load_ckpt(P.load_path)

checkpoint = model_state_dict

missing, unexpected = model.load_state_dict(checkpoint, strict=not P.no_strict) 
print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

assert P.eval_episode == 0 or P.use_paco_stats or P.replay_buffer_size > 0, "Cannot perform eval in this condition!"

if P.multi_gpu:
    model = apex.parallel.convert_syncbn_model(model)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)


### Initialize data ###
source_datasets_test = get_dataset_2(P, episode=P.eval_episode, train=False)
if P.replay_buffer_size > 0 and P.eval_episode > 0:
    assert 'selected_ids' in aux_data, "Replay buffer data is not present in the checkpoint"
    selected_replay_ids = aux_data["selected_ids"]
    source_datasets_replay = get_replay_datasets(P, episode=P.eval_episode, selected_ids_dict=selected_replay_ids)
    source_datasets_test.extend(source_datasets_replay)

whole_source_test = ConcatDataset(source_datasets_test)

known_tgt_ds, unknown_tgt_ds = get_test_datasets(P, episode=P.eval_episode)


if not P.multi_gpu:
    source_test_loader = DataLoader(whole_source_test, shuffle=False, batch_size=P.batch_size, num_workers=P.num_workers)
    known_tgt_loader = DataLoader(known_tgt_ds, shuffle=False, batch_size=1, num_workers=P.num_workers)
    unknown_tgt_loader = DataLoader(unknown_tgt_ds, shuffle=False, batch_size=1, num_workers=P.num_workers)

else:
    source_test_sampler = DistributedSampler(whole_source_test, num_replicas=P.n_gpus, rank=P.local_rank)
    source_test_loader = DataLoader(whole_source_test, sampler=source_test_sampler, shuffle=False, batch_size=P.batch_size, num_workers=P.num_workers)

    known_tgt_sampler = DistributedSampler(known_tgt_ds, num_replicas=P.n_gpus, rank=P.local_rank)
    known_tgt_loader = DataLoader(known_tgt_sampler, sampler=known_tgt_sampler, shuffle=False, batch_size=1, num_workers=P.num_workers)

    unknown_tgt_sampler = DistributedSampler(unknown_tgt_ds, num_replicas=P.n_gpus, rank=P.local_rank)
    unknown_tgt_loader = DataLoader(unknown_tgt_sampler, sampler=unknown_tgt_sampler, shuffle=False, batch_size=1, num_workers=P.num_workers)



