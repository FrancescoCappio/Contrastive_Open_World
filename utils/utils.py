import os
import pickle
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from utils.dist_utils import get_rank


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True, local_rank=0):
        self.local_rank = local_rank
        if self.local_rank == 0:
            if not os.path.exists("./logs/"):
                os.mkdir("./logs/")

            logdir = self._make_dir(fn)
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            if len(os.listdir(logdir)) != 0 and ask:
                ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)

            self.set_dir(logdir)

    def _make_dir(self, fn):
        today = datetime.today().strftime("%y%m%d")
        logdir = 'logs/' + fn
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        if self.local_rank == 0:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.local_rank == 0:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.local_rank == 0:
            #self.writer.add_scalar(tag, value, step)
            pass

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        if self.local_rank == 0:
            #self.writer.add_images(tag, images, step)
            pass

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        if self.local_rank == 0:
            #self.writer.add_histogram(tag, values, step, bins='auto')
            pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

def log_or_print(string, logger=None):
    if logger:
        logger.log(string)
    else:
        if get_rank() == 0:
            print(string)

def check_resume(logdir):
    logdir = f"logs/{logdir}"
    last_checkpoint_txt_file = f"{logdir}/last_checkpoint.txt"
    if not os.path.isfile(last_checkpoint_txt_file):
        return False
    with open(last_checkpoint_txt_file, "r") as fin:
        name = fin.read().strip()
    ckpt_path = logdir + "/" + name
    return os.path.isfile(ckpt_path)

def load_ckpt(ckpt_path=None, logdir=None, logger=None):

    assert ckpt_path or logdir, "You have to specify a model_path or a log directory from which to resume"

    # load last checkpoint
    if not ckpt_path:

        last_checkpoint_txt_file = f"{logdir}/last_checkpoint.txt"
        assert os.path.isfile(last_checkpoint_txt_file), "last checkpoint not known"
        resume = True
        with open(last_checkpoint_txt_file, "r") as fin:
            name = fin.read().strip()
        ckpt_path = logdir + "/" + name
        try: 
            last_it = int(name.split(".")[0])
        except ValueError:
            last_it = None
    else:
        name = os.path.basename(ckpt_path)
        resume = False
        try:
            last_it = int(name.split(".")[0])
        except ValueError:
            last_it = None

    assert os.path.isfile(ckpt_path), f"File {ckpt_path} does not exist"
    log_or_print(f"Loading checkpoint from {ckpt_path}", logger)

    ckpt = torch.load(ckpt_path,map_location='cpu')

    # check if the checkpoint contains meta-data or if it is only a model state dict
    if "model" in ckpt:
        model_state_dict = ckpt["model"]

        # if source info are already in checkpoint the eval computation is easier
        if "aux_data" in ckpt:
            aux_data = ckpt["aux_data"]
        else:
            aux_data = None
        
        if last_it is None and "current_it" in ckpt: 
            last_it = ckpt["current_it"]

    else:
        assert not resume, "Resume not supported from this checkpoint!"

        model_state_dict = ckpt
        scheduler_state_dict, optimizer_state_dict = None, None
        aux_data = None

    log_or_print("Checkpoint loaded", logger)
    return model_state_dict, aux_data, last_it

def save_ckpt(its, logdir, model_state, aux_data: dict = None, logger=None, save_all=False, name=None):

    # is name is None we use current it vale as name
    file_name = f"{its}.model"
    if name is not None: 
        file_name = f"{name}.model"
    
    output_file_name = f"{logdir}/{file_name}"
    last_checkpoint_txt_file = f"{logdir}/last_checkpoint.txt"

    # if not save all, delete last checkpoint
    if os.path.isfile(last_checkpoint_txt_file) and not save_all:
        # get name of last checkpoint
        with open(last_checkpoint_txt_file, "r") as fin:
            name = fin.read().strip()
        ckpt_path = logdir + "/" + name
        # remove last checkpoint txt file and corresponding checkpoint
        os.remove(last_checkpoint_txt_file)
        os.remove(ckpt_path)

    ckpt_data = {}
    ckpt_data["model"] = model_state

    if aux_data:
        ckpt_data["aux_data"] = aux_data

    ckpt_data["current_it"] = its

    torch.save(ckpt_data, output_file_name)
    with open(last_checkpoint_txt_file, "w") as fout:
        fout.write(file_name)
    log_or_print(f"Checkpoint saved in {output_file_name}", logger)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def normalize_images(P, inputs):
    mean = torch.tensor(P.im_mean).to(inputs.device)
    std = torch.tensor(P.im_std).to(inputs.device)
    return  ((inputs.permute(0,2,3,1)-mean)/std).permute(0,3,1,2)

def denormalize_images(P, inputs):
    mean = torch.tensor(P.im_mean).to(inputs.device)
    std = torch.tensor(P.im_std).to(inputs.device)
    return  (inputs.permute(0,2,3,1)*std+mean).permute(0,3,1,2)

