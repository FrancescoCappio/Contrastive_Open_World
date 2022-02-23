from abc import *
import torch
import torch.nn as nn

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, simclr_dim=128):
        super(BaseModel, self).__init__()
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )

        # PaCo style parametric centers:
        prototypes = torch.randn((num_classes,simclr_dim))
        self.prototypes = nn.Parameter(data=prototypes, requires_grad=True)

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs, penultimate=False, simclr=False, prototypes=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)

        if prototypes:
            _return_aux = True
            _aux['prototypes'] = self.prototypes

        if _return_aux:
            return None, _aux

        return None
