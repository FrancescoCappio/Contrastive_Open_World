from common.eval import *
from tqdm import tqdm
import numpy as np

model.eval()

if aux_data is None: 
    if P.eval_episode == 0:
        P.prog_to_real = {}
        P.real_to_prog = {}

        for p_lbl, r_lbl in zip(range(P.total_n_classes), P.all_learned_classes):
            P.prog_to_real[p_lbl] = r_lbl
            P.real_to_prog[r_lbl] = p_lbl
    else:
        raise RuntimeError("Cannot perform real to prog mapping if eval_episode > 0")
else:
    P.prog_to_real = aux_data['prog_to_real']
    P.real_to_prog = aux_data['real_to_prog']
    P.online_class_compactness = aux_data['avg_class_compactness']


if P.mode == "OWR_eval":
    from evals.evals import OWR_eval

    with torch.no_grad():
        OWR_eval(P, model, source_test_loader, known_tgt_loader, unknown_tgt_loader,logger=None)

elif P.mode == "openset_eval":
    from evals.evals import openset_eval

    with torch.no_grad():
        openset_eval(P, model, source_test_loader, target_test_loader, logger=None,source_stats=source_stats)

elif P.mode == "openset_eval_save":
    from evals.evals import openset_eval

    with torch.no_grad():
        openset_eval(P, model, source_test_loader, target_test_loader, logger=None, source_stats=source_stats, save_dir=P.save_dir)


elif P.mode == "eval_known_selection":
    from evals.evals import compute_confident_known_mask

    with torch.no_grad():
        known_mask, known_pseudo_labels, known_gt_labels = compute_confident_known_mask(P, model, source_test_loader, target_test_loader, logger=None)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(known_gt_labels[known_mask], known_pseudo_labels[known_mask])

    gt_known = 0

    known_gt_lbls = known_gt_labels[known_mask]
    number_real_known = len(known_gt_lbls[known_gt_lbls < P.n_classes])
    percentage_true_known = number_real_known/len(known_mask.nonzero()[0])
    print("Selected {} target samples as known. Classification accuracy: {:.4f}. Percentage of gt known: {:.4f}".format(len(known_mask.nonzero()[0]), acc, percentage_true_known))
else:
    raise NotImplementedError()



