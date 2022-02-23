import time
import itertools
import math
import os

import diffdist.functional as distops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import models.transform_layers as TL
from utils.temperature_scaling import _ECELoss
from utils.utils import AverageMeter, set_random_seed, normalize
from tqdm import tqdm
import sys
from utils.dist_utils import synchronize, all_gather
from utils.utils import log_or_print, normalize
np.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

def get_features(P, model, test_loaders, to_normalize=True, layer="simclr", distributed=False):
    model.eval()

    feats = []
    labels = []
    paths = []
    ids = []
    out_dict = {}
    
    for loader in test_loaders:
        for batch in tqdm(loader):
            images, lbls, img_ids, orig_paths = batch
            images = images.to(device)
            with torch.no_grad():
                _, output_aux = model(images, penultimate=True, simclr=True)
    
            out_feats = output_aux[layer]
            if to_normalize:
                out_feats = normalize(out_feats)
            for feat, lbl, img_id, orig_path in zip(out_feats, lbls, img_ids, orig_paths):
                labels.append(lbl.item())
                paths.append(orig_path)
                cpu_feat = feat.to(cpu_device)
                feats.append(cpu_feat.unsqueeze(0))
                ids.append(img_id.item())
                out_dict[img_id.item()] = {'feat': cpu_feat, 'label': lbl.item(), 'path': orig_path}

    feats=torch.cat(feats)

    if distributed and P.n_gpus > 1:
        raise NotImplementedError("This does not work anymore with concatdataset of class datasets as img_ids are per-class")
    return feats, np.array(labels), paths, ids

def rescale_cosine_similarity(similarities):
    return (similarities+1)/2

def compute_source_prototypes(P, model, source_loader, eval_layer, distributed=False, logger=None):
    
    # we extract features for all source samples
    source_feats, source_gt_labels, paths, ids = get_features(P,model, [source_loader], layer=eval_layer, to_normalize=True, distributed=distributed)

    # compute source general prototype, norm, compactness
    src_prt = source_feats.mean(dim=0)
    # project on hypersphere
    src_prt_norm = src_prt.norm()
    src_prt = src_prt/src_prt_norm
    # compute average tgt samples distance from tgt prt
    total = 0
    for src_feat in source_feats:
        total += (src_feat*src_prt).sum()
    log_or_print(f"Source prototype norm {src_prt_norm}: Source compactness: {total/len(source_feats)}", logger=logger)

    # we compute prototypes for source classes using source data
    labels_set = set(source_gt_labels.tolist())
    prototypes = {}
    for label in labels_set:
        lbl_mask = source_gt_labels == label
        source_this_label = source_feats[lbl_mask]
        prototypes[label] = source_this_label.mean(dim=0)
        prototypes[label] = prototypes[label]/prototypes[label].norm()

    hyp_prototypes = np.stack([prt for prt in prototypes.values()])
    # we also compute average distance between nearest prototypes
    topk_sims = np.zeros((len(hyp_prototypes)))
    for idx, hyp_prt in enumerate(hyp_prototypes):
        similarities = (hyp_prt*hyp_prototypes).sum(1)
        similarities = rescale_cosine_similarity(similarities)
        similarities.sort()
        topk_val = similarities[-2]
        topk_sims[idx] = topk_val

    # let's compute average cluster compactness
    # we will need a threshold which will be based on this value 
    cls_compactness_tot = 0
    cls_compactness_vector = []

    idx_ordered_nearest_prototypes_by_class = {}

    for cls in prototypes.keys():
        cls_prototype = prototypes[cls]

        idx_ordered_nearest_prototypes_by_class[cls] = []
        similarity_by_class = []
        
        source_this_label = source_feats[source_gt_labels == cls]
        idx_this_label = np.array(ids)[source_gt_labels == cls]

        if len(source_this_label) == 0:
            continue

        tot_similarities = 0

        for src_feat in source_this_label:
            similarity = (src_feat*cls_prototype).sum()
            similarity = rescale_cosine_similarity(similarity)
            tot_similarities += similarity.item()

            similarity_by_class.append(similarity.item())

        avg_cls_similarity = tot_similarities/len(source_this_label)
        cls_compactness_tot += avg_cls_similarity
        cls_compactness_vector.append(avg_cls_similarity)

        # from the lowest value to the highest value, from the less similar to the most similar
        perm_ordered = np.array(similarity_by_class).argsort()
        idx_ordered_nearest_prototypes_by_class[cls] = idx_this_label[perm_ordered]

    cls_compactness_vector = torch.tensor(cls_compactness_vector)

    return source_feats, prototypes, cls_compactness_vector, topk_sims, idx_ordered_nearest_prototypes_by_class

def compute_threshold_multiplier(avg_compactness, avg_cls_dist):
    y = (1-avg_cls_dist)
    x = (1-avg_compactness)
    z = y/(2*x)
    f_z = math.log(z) + 1
    k = 1
    d = 5
    try:
        h_z = math.exp(k*(z-d)) / (1 + math.exp(k*(z-d))) + 1
    except:
        h_z = 2
    nr = f_z*h_z
    return nr

def compute_source_stats(P, model, source_loader, eval_layer="simclr", logger=None):
    """
    The purpose of this method is to perform a forward of source data in order to compute 
    source prototypes positions, class compactness and separation 
    """
    # check if distributed
    distributed = isinstance(source_loader.sampler, torch.utils.data.distributed.DistributedSampler)
    model.eval()

    # compute source prototypes and compactness for known classes
    _, hyp_prototypes, cls_compactness_vector, topk_sims, idx_ordered_nearest_prototypes_by_class = compute_source_prototypes(P, model, source_loader, eval_layer, distributed=distributed, logger=logger)
    avg_min_sim = topk_sims.mean()

    cls_compactness_avg = cls_compactness_vector.mean()
    log_or_print(f"Class compactness avg: {cls_compactness_avg}, std: {cls_compactness_vector.std()} . Avg_min_sim: {avg_min_sim}, std: {topk_sims.std()}", logger)

    return {"source_prototypes": hyp_prototypes, 
            "compactness": cls_compactness_avg,
            "separation": avg_min_sim}, idx_ordered_nearest_prototypes_by_class

def eval_preliminaries(P, model, source_loader, tgt_known_loader, tgt_unknown_loader, eval_layer="simclr", source_stats=None, logger=None):
    """
    All eval procedures requires to gather source stats and target features
    """
    if source_stats is None:
        source_stats,_ = compute_source_stats(P, model, source_loader, eval_layer, logger)
    # check if distributed
    distributed = isinstance(source_loader.sampler, torch.utils.data.distributed.DistributedSampler)
    model.eval()

    # now we get features for target samples 
    tgt_known_feats, tgt_known_labels, tgt_known_paths,_ = get_features(P, model, [tgt_known_loader], layer=eval_layer, to_normalize=True, distributed=distributed)
    tgt_unknown_feats, tgt_unknown_labels, tgt_unknown_paths,_ = get_features(P, model, [tgt_unknown_loader], layer=eval_layer, to_normalize=True, distributed=distributed)

    # compute whole target prototype
    target_feats = torch.cat((tgt_known_feats, tgt_unknown_feats))
    tgt_prt = target_feats.mean(dim=0)
    # project on hypersphere
    tgt_prt_norm = tgt_prt.norm()
    tgt_prt = tgt_prt/tgt_prt_norm
    # compute average tgt samples distance from tgt prt
    total = 0
    for tgt_feat in target_feats:
        total += (tgt_feat*tgt_prt).sum()
    log_or_print(f"Target prototype norm {tgt_prt_norm}: Target compactness: {total/len(target_feats)}", logger=logger)

    # compute unknown targets prototypes norm
    uk_target_feats = tgt_unknown_feats
    # compute target prototype
    uk_tgt_prt = uk_target_feats.mean(dim=0)
    # project on hypersphere
    uk_tgt_prt_norm = uk_tgt_prt.norm()
    uk_tgt_prt = uk_tgt_prt/uk_tgt_prt_norm
    # compute average tgt samples distance from tgt prt
    total = 0
    for uk_tgt_feat in uk_target_feats:
        total += (uk_tgt_feat*uk_tgt_prt).sum()
    log_or_print(f"Target UNK prototype norm {uk_tgt_prt_norm}: Target UNK compactness: {total/len(uk_target_feats)}", logger=logger)

    return source_stats, tgt_known_feats, tgt_unknown_feats, tgt_known_labels, tgt_unknown_labels, tgt_known_paths, tgt_unknown_paths

def compute_known_threshold(P, source_stats, train=False):

    alpha_multiplier = P.eval_alpha_multiplier
    cls_compactness = source_stats["compactness"]
    cls_separation = source_stats["separation"]
    threshold_multiplier = compute_threshold_multiplier(cls_compactness, cls_separation) * alpha_multiplier
    known_threshold = (1-cls_compactness) * threshold_multiplier
    return known_threshold

def compute_confident_known_mask(P, model, source_loader, target_loader, logger, eval_layer="simclr", source_stats=None, target_feats=None, target_gt_labels=None):
    """
    for each target sample we measure distance from nearest prototype. If ditance is lower than a threshold we select 
    this sample and the prototype label as pseudo label 
    """
    if source_stats is None:
        source_stats, target_feats, target_gt_labels, _ = eval_preliminaries(P, model, source_loader, target_loader, eval_layer, source_stats=source_stats, logger=logger)
    source_prototypes = source_stats["source_prototypes"]

    known_threshold = compute_known_threshold(P, source_stats, train=True)
    
    # prepare vectors to store mask and labels:
    known_mask = np.zeros((len(target_feats)), dtype=np.bool)
    known_pseudo_labels = P.n_classes*np.ones((len(target_feats)), dtype=np.uint32)
    known_gt_labels = P.n_classes*np.ones((len(target_feats)), dtype=np.uint32)

    # for each target element, evaluate normality 
    for idx, (tgt_feat, tgt_gt_label) in enumerate(zip(target_feats, target_gt_labels)): 

        similarities = (tgt_feat*source_prototypes).sum(dim=1)
        similarities = rescale_cosine_similarity(similarities)

        highest = similarities.max()
        cls_id = similarities.argmax()

        # check whether it is near enough to nearest prototype to be considered known
        if highest >= (1 - known_threshold):
            known_mask[idx] = True
            known_pseudo_labels[idx] = cls_id

        known_gt_labels[idx] = tgt_gt_label.item()
        if tgt_gt_label > P.n_classes: # unknown class
            known_gt_labels[idx] = P.n_classes

    return known_mask, known_pseudo_labels, known_gt_labels

def openset_eval(P, model, source_loader, target_loader, logger, eval_layer="simclr", save_dir=None, source_stats=None):
    """
    Compute openset eval metrics
    """

    log_or_print("Running openset eval", logger)
    source_stats, target_feats, target_gt_labels, paths = eval_preliminaries(P, model, source_loader, target_loader, eval_layer, source_stats=source_stats, logger=logger)
    source_prototypes = source_stats["source_prototypes"]

    # define counters we need for openset eval
    samples_per_class = np.zeros(P.n_classes + 1)
    correct_pred_per_class = np.zeros(P.n_classes + 1)
    closed_set_correct_pred_per_class = np.zeros(P.n_classes + 1)

    # for each target sample we have to make a predictions. So we compare it with all the prototypes. 
    # the sample is associated with the class of the nearest prototype if its similarity with this prototype 
    # is higher than a certain threshold
    normality_threshold = compute_known_threshold(P, source_stats, train=False)

    # prepare tensors for masks
    correct_known_mask = torch.zeros(target_feats.shape[0],dtype=bool)
    predicted_known_mask = torch.zeros(target_feats.shape[0],dtype=bool)
    known_normality_scores = []
    unknown_normality_scores = []

    predictions = []

    for tgt_idx, (tgt_feat, tgt_gt_label, path) in enumerate(zip(target_feats, target_gt_labels, paths)): 

        similarities = (tgt_feat*source_prototypes).sum(dim=1)
        similarities = rescale_cosine_similarity(similarities)

        highest = similarities.max()
        cls_id = similarities.argmax()

        # extract from tuple
        path = path[0]
        
        predictions.append(f"./{path} {cls_id} {1-highest}")

        if cls_id == tgt_gt_label:
            closed_set_correct_pred_per_class[cls_id] += 1

        # check whether it is near enough to nearest prototype to be considered known
        if highest < (1 - normality_threshold):
            # this is the unknown cls_id
            cls_id = P.n_classes

        # accumulate prediction
        if tgt_gt_label >= P.n_classes:
            # unknown gt
            tgt_gt_label = P.n_classes
            unknown_normality_scores.append(highest)
        else:
            # known gt
            correct_known_mask[tgt_idx] = True
            known_normality_scores.append(highest)
            
        samples_per_class[tgt_gt_label] += 1
        predicted_known_mask[tgt_idx] = cls_id < P.n_classes
        if cls_id == tgt_gt_label:
            correct_pred_per_class[cls_id] += 1

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir + '/target_feats.npy',target_feats.numpy())
        np.save(save_dir + '/target_known_gt.npy',correct_known_mask.numpy())
        np.save(save_dir + '/target_known_predicted.npy',predicted_known_mask.numpy())
        np.save(save_dir + '/prototypes.npy', source_prototypes)
        with open(save_dir + '/predictions.txt', "w") as out_file:
            for l in predictions:
                out_file.write(l+"\n")

    acc_os_star = np.mean(correct_pred_per_class[0:len(correct_pred_per_class)-1] / samples_per_class[0:len(correct_pred_per_class)-1])
    acc_unknown = (correct_pred_per_class[-1] / samples_per_class[-1])
    acc_hos = 2 * (acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)
    acc_os = np.mean(correct_pred_per_class/ samples_per_class)
    avg_known_accuracy = 100*(closed_set_correct_pred_per_class[0:len(correct_pred_per_class)-1].sum() / samples_per_class[0:len(correct_pred_per_class)-1].sum())

    acc_os *= 100
    acc_os_star *= 100
    acc_unknown *= 100
    acc_hos *= 100

    known_scores = torch.stack(known_normality_scores)
    unknown_scores = torch.stack(unknown_normality_scores)
    auroc=100*(get_auroc(known_scores, unknown_scores))

    log_or_print('[AUROC %6f]' %(auroc), logger)
    log_or_print('[OS %6f]' %(acc_os), logger)
    log_or_print('[OS* %6f]' % (acc_os_star), logger)
    log_or_print('[UNK %6f]' % (acc_unknown), logger)
    log_or_print('[HOS %6f]' % (acc_hos), logger)
    log_or_print('[CS_ACC %6f]' % (avg_known_accuracy), logger)

    return source_stats, target_feats, target_gt_labels

def OWR_eval(P, model, source_loader, tgt_known_loader, tgt_unknown_loader, logger, eval_layer="simclr", source_stats=None):
    """
    Compute openset eval metrics
    """

    log_or_print("Running OWR eval", logger)
    source_stats, tgt_known_feats, tgt_unknown_feats, tgt_known_labels, tgt_unknown_labels, _, _ = \
            eval_preliminaries(P, model, source_loader, tgt_known_loader, tgt_unknown_loader, eval_layer, source_stats=source_stats, logger=logger)

    source_prototypes = source_stats["source_prototypes"]
    prts_labels, prts = [], []
    for lbl, prt in source_prototypes.items():
        prts_labels.append(lbl)
        prts.append(prt)
    source_prototypes=torch.stack(prts)

    # define counters we need for openset eval
    samples_per_class = np.zeros(P.n_classes + 1)
    correct_pred_per_class = np.zeros(P.n_classes + 1)
    closed_set_correct_pred_per_class = np.zeros(P.n_classes + 1)

    # for each target sample we have to make a predictions. So we compare it with all the prototypes. 
    # the sample is associated with the class of the nearest prototype if its similarity with this prototype 
    # is higher than a certain threshold
    normality_threshold = compute_known_threshold(P, source_stats, train=False)

    known_scores = torch.zeros(len(tgt_known_labels), dtype=torch.float)
    known_cs_preds_no_rej = np.zeros_like(tgt_known_labels)
    known_cs_preds_rej = np.zeros_like(tgt_known_labels)
    known_correct_no_rej = 0
    known_correct_rej = 0

    for idx, (tgt_feat, tgt_gt_label) in enumerate(zip(tgt_known_feats, tgt_known_labels)): 
        similarities = (tgt_feat*source_prototypes).sum(dim=1)
        similarities = rescale_cosine_similarity(similarities)

        highest = similarities.max()
        predicted_label = prts_labels[similarities.argmax()]

        closed_set_pred_no_rej = predicted_label
        closed_set_pred_rej = closed_set_pred_no_rej

        # if the similarity to the nearest prototype is not high enough we reject the sample
        if highest < (1 - normality_threshold):
            # this is the unknown cls_id
            closed_set_pred_rej = -1

        known_scores[idx] = highest
        known_cs_preds_no_rej[idx] = closed_set_pred_no_rej
        known_cs_preds_rej[idx] = closed_set_pred_rej
        if closed_set_pred_no_rej == tgt_gt_label.item():
            known_correct_no_rej += 1
        if closed_set_pred_rej == tgt_gt_label.item():
            known_correct_rej += 1

    unknown_scores = torch.zeros(len(tgt_unknown_labels), dtype=torch.float)
    unknown_correct_preds = 0
    for idx, (tgt_feat, tgt_gt_labl) in enumerate(zip(tgt_unknown_feats, tgt_unknown_labels)):
        similarities = (tgt_feat*source_prototypes).sum(dim=1)
        similarities = rescale_cosine_similarity(similarities)

        highest = similarities.max()
        # if the similarity to the nearest prototype is not high enough we reject the sample
        if highest < (1 - normality_threshold):
            # this is the unknown cls_id
            unknown_correct_preds += 1
        unknown_scores[idx] = highest

    auroc = get_auroc(known_scores, unknown_scores)

    cs_acc_no_rej = known_correct_no_rej / len(tgt_known_labels)
    cs_acc_rej = known_correct_rej / len(tgt_known_labels)
    unk_acc = unknown_correct_preds / len(tgt_unknown_labels)

    hos = 2 * (cs_acc_rej * unk_acc) / (cs_acc_rej + unk_acc)

    log_or_print('[AUROC   %6f]' % (100 * auroc), logger)
    log_or_print('[CS_NO_R %6f]' % (100 * cs_acc_no_rej), logger)
    log_or_print('[CS_R    %6f]' % (100 * cs_acc_rej), logger)
    log_or_print('[UNK     %6f]' % (100 * unk_acc), logger)
    log_or_print('[HOS     %6f]' % (100 * hos), logger)

    log_or_print('{},{},{},{},{}'.format(100*auroc, 100*cs_acc_no_rej, 100*cs_acc_rej, 100*unk_acc, 100*hos), logger)

    return


def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)
