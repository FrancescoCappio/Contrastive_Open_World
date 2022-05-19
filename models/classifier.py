import torch.nn as nn
import torch

from models.resnet_imagenet import resnet50, resnet101, resnet18
import models.transform_layers as TL

def get_simclr_augmentation(P, image_size):

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
    )

    return transform

def get_simclr_augmentation_identity():
    # disable resize crop
    resize_crop = nn.Identity()
    # Transform define #
    transform = nn.Sequential(resize_crop)
    return transform

def get_classifier(mode, n_classes=10, pretrain=None,simclr_dim=128):
    if mode in ('resnet50_imagenet', 'resnet101', 'resnet18'):
        if '50' in mode:
            classifier = resnet50(num_classes=n_classes,simclr_dim=simclr_dim)
        elif '101' in mode: 
            classifier = resnet101(num_classes=n_classes,simclr_dim=simclr_dim)
        elif '18' in mode:
            classifier = resnet18(num_classes=n_classes,simclr_dim=simclr_dim)
    elif "deit" in mode:
        from models.vit_models import deit_base
        if mode == "deit_base":
            classifier = deit_base(simclr_dim=simclr_dim)
        else:
            raise NotImplementedError(f"Model {mode} not known")
    else:
        raise NotImplementedError(f"Model {mode} not known")

    if not pretrain is None:
        ckpt = torch.load(pretrain)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        missing, unexpected = classifier.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {pretrain}")
        print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    return classifier

