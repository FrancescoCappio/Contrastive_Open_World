import torch
import torch.nn as nn
import timm

from models.base_model import BaseModel
from models.transform_layers import NormalizeLayer

class ViTModel(BaseModel):
    def __init__(self, model_name="deit_base_patch16_224", simclr_dim=128):
        # by passing num_classes=0 we obtain a model without the last linear classification layer, we don't need it!
        vit_model = timm.create_model(model_name, pretrained=True,num_classes=0)
        last_dim = vit_model.embed_dim 
        super().__init__(last_dim=last_dim, simclr_dim=simclr_dim)
        self.vit_model = vit_model
        self.normalize = NormalizeLayer()

    def penultimate(self, x, all_features=False):
        return self.vit_model(self.normalize(x))

def deit_base(**kwargs):
    return ViTModel(model_name='deit_base_patch16_224', **kwargs)
