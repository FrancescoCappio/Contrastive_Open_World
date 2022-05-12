from utils.utils import Logger
from utils.utils import save_ckpt, load_ckpt, check_resume, normalize
from utils.utils import AverageMeter
from evals.evals import compute_confident_known_mask, openset_eval, compute_source_stats, get_features
from datasets.datasets import get_dataset_2, Subset, ConcatDataset, get_replay_datasets
import time
import sys
import math
import numpy as np
from sklearn.metrics import accuracy_score

from common.train import *
import torch
import wandb 

from training.sup import setup

def exit_handler():
    print("Exiting")

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    logger.log(f"Total Trainable Params: {total_params}")
    return total_params

def save(logger, P, its, model, episode, selected_ids=None):
    model_state = model.state_dict()

    aux_data = get_aux_data(P)
    if selected_ids is not None:
        aux_data["selected_ids"] = selected_ids
    save_ckpt(its, logger.logdir, model_state, aux_data, logger, P.save_all, name=f"episode_{episode}")

def get_aux_data(P):
    aux_data = {}

    aux_data['prog_to_real'] = P.prog_to_real
    aux_data['real_to_prog'] = P.real_to_prog

    aux_data['avg_class_compactness'] = P.online_class_compactness
    return aux_data

def log_iter(meters, current_lr, its):
    # log training stats on stdout 
    eta_sec = (P.iterations - its) * meters['time'].average
    hour = eta_sec // 3600
    eta_sec = eta_sec % 3600
    eta_min = eta_sec // 60
    eta_sec = eta_sec % 60

    log_string = '[Iteration %3d] [Avg time %.2fs] [ETA %02dh%02dm%02ds] [LR %.6f]' % (its, meters["time"].average, hour, eta_min, eta_sec, current_lr)
    log_data = {
        "iter_time": meters["time"].average,
        "lr": current_lr,
        }
    for k in meters.keys():
        if not k == "time":
            log_string += ' [{} {:6.4f}]'.format(k, meters[k].average)
            log_data[k] = meters[k].average
            meters[k] = AverageMeter()
    logger.log(log_string)

    wandb.log(log_data,step=its)

def periodic_source_eval(its):
    # we perform periodic *evals* episodes in which we simply compute source statistics and print them 

    ### Initialize test data ###
    source_datasets_test = get_dataset_2(P, episode=P.current_episode, train=False)
    if P.replay_buffer_size > 0 and P.current_episode > 0:
        test_replay_ids = replay_selected_ids
        source_datasets_replay = get_replay_datasets(P, episode=P.current_episode, selected_ids_dict=test_replay_ids)
        source_datasets_test.extend(source_datasets_replay)
    whole_source_test = ConcatDataset(source_datasets_test)
    source_test_loader = DataLoader(whole_source_test, shuffle=False, batch_size=P.batch_size, num_workers=P.num_workers)

    ### Compute and log source stats ###
    model.eval()
    source_stats,idx_ordered_nearest_prototypes_by_class = compute_source_stats(P, model, source_test_loader, eval_layer='simclr', logger=logger)
    compactness = source_stats["compactness"].item()
    separation = source_stats["separation"]
    alpha_ratio = (1-separation)/(2*(1-compactness))
    model.train()

    ### Check if source stats match compactness-separation constraint for this episode ###
    if alpha_ratio > 1 + P.compactness_margin:
        P.stats_condition_matched = True
    logger.log(f"Alpha ratio: {alpha_ratio}. Termination condition matched: {P.stats_condition_matched}")

    wandb.log({"alpha_ratio": alpha_ratio, "compactness": compactness, "separation": separation},step=its)

    return idx_ordered_nearest_prototypes_by_class

import atexit
atexit.register(exit_handler)

P.save_all = True
P.online_class_compactness = None
# setup training routine
train, fname = setup(P, model, source_test_loader)

logger = Logger(fname, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

wandb.init(project="COW", name=fname, config=P)

P.prog_to_real = {}
P.real_to_prog = {}

for p_lbl, r_lbl in zip(range(P.total_n_classes), P.all_learned_classes):
    P.prog_to_real[p_lbl] = r_lbl
    P.real_to_prog[r_lbl] = p_lbl

# Run experiments
losses = dict()
losses['time'] = AverageMeter()
check = time.time()

data_iter = iter(train_loader)
count_parameters(model)

logger.log("Learning classes: " + str(P.classes_current_episode))

replay_selected_ids = None

P.current_episode = 0
its = 1
current_ep_its = 1
P.stats_condition_matched = False
current_ep_min_its = P.ep_0_min_its

while True: 
    model.train()

    # train one iteration
    meters_dict, train_loader, data_iter = train(P, its, model, criterion, optimizer, scheduler_warmup, train_loader, data_iter, logger=logger, simclr_aug=simclr_aug)

    model.eval()
    # log
    for k in meters_dict.keys():
        if k not in losses:
            losses[k] = AverageMeter()
        losses[k].update(meters_dict[k],1)
    losses['time'].update(time.time()-check, 1)

    check = time.time()
    if its%100 == 0:
        lr = optimizer.param_groups[0]['lr']
        log_iter(losses, lr, its)

    if its%2500 == 0:
        # save current model 
        if P.local_rank == 0:
            save(logger, P, its, model, episode=P.current_episode, selected_ids=replay_selected_ids)

        idx_ordered_nearest_prototypes_by_class = periodic_source_eval(its)

    # check if we match conditions to end current episode
    if current_ep_its > current_ep_min_its and P.stats_condition_matched:
        logger.log(f"End of episode {P.current_episode}")
        # we start a new learning episode
        P.stats_condition_matched = False
        current_ep_its = 0
        current_ep_min_its = P.eps_min_its

        # first of all we save the current model, with ids selected and used for replay buffer
        if P.local_rank == 0:
            save(logger, P, its, model, episode=P.current_episode, selected_ids=replay_selected_ids)
        # if this is the last episode training should end here
        if P.current_episode == P.total_episodes - 1:
            logger.log("End of last episode")
            break

        # if necessary we store a number of samples from old classes in a replay buffer: 
        old_class_datasets = []
        replay_selected_ids = {}
        if P.replay_buffer_size > 0:
            # we keep a subset of the old dataset as a replay buffer

            if P.replay_selection_strategy == 'random':

                n_classes = len(P.all_learned_classes)
                samples_per_class = math.floor(P.replay_buffer_size / n_classes)
                for ds in train_loader.dataset.datasets:
                    all_ids = np.arange(len(ds))
                    if len(all_ids) <= samples_per_class:
                        old_class_datasets.append(ds)
                        selected_ids = all_ids
                    else:
                        selected_ids = np.random.choice(all_ids, size=samples_per_class, replace=False)
                        old_class_datasets.append(ConcatDataset([Subset(ds, indices=selected_ids)]))
                    # we should save the list of selected ids for future evals
                    lbl = ds[0][1]
                    replay_selected_ids[lbl] = selected_ids

            else:
                raise NotImplementedError("Unknown replay selection strategy: {P.replay_selection_strategy}")

        # we create the dataset with the new episode classes
        P.current_episode += 1
        train_sets = get_dataset_2(P, episode=P.current_episode)
        train_sets.extend(old_class_datasets)
        logger.log("Starting a new episode")
        logger.log("Learning classes: " + str(P.classes_current_episode))
        whole_source = ConcatDataset(train_sets)
        my_sampler = BalancedMultiSourceRandomSampler(whole_source, P.batch_p, P.local_rank, P.n_gpus)
        print(f"Rank {P.local_rank}: sampler_size: {len(my_sampler)}. Dataset_size: {len(whole_source)}")

        if P.adapt_batch_K:
            batch_K = len(P.all_learned_classes)
            single_GPU_batch_K = batch_K/P.n_gpus
            single_GPU_batch_size = int(P.batch_p*single_GPU_batch_K)

            if P.adapt_lr:
                base_lr = P.lr_init # this is used for batch_K equal to P.batch_K 
                adapted_lr = (batch_K/P.batch_K) * base_lr
                base_optimizer = optim.SGD(model.parameters(), lr=adapted_lr, momentum=0.9, weight_decay=P.weight_decay)
                optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
                lr_decay_gamma = 0.1

                # our scheduler is warmup + constant (obtained via step lr without steps)
                scheduler_warmup = ConstantScheduler(optimizer)

        train_loader = DataLoader(whole_source, sampler=my_sampler, batch_size=single_GPU_batch_size, **kwargs)
        data_iter = iter(train_loader)

        # we should assign new classes to existing progressive class labels
        prog_to_real = {}
        real_to_prog = {}
        
        # previous episodes mappings should be maintained
        prog_to_real.update(P.prog_to_real)
        real_to_prog.update(P.real_to_prog)
        
        last_progressive_idx = max(P.real_to_prog.values())
        if P.new_classes_mapping == 'random':
            logger.log("Selected random association between old prototypes and new classes")
            for r_lbl in P.classes_current_episode:
                last_progressive_idx += 1
                real_to_prog[r_lbl] = last_progressive_idx
                prog_to_real[last_progressive_idx] = r_lbl
                logger.log(f"\tClass {r_lbl} matched with prototype {last_progressive_idx}")
        else:
            raise NotImplementedError(f"Unknown class mapping strategy: {P.new_classes_mapping}")

        P.real_to_prog = real_to_prog
        P.prog_to_real = prog_to_real

    ### Count all and current episode iters ###
    its += 1
    current_ep_its += 1 
