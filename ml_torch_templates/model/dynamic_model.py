# %%
# 2. implement model architecture

import torch.nn as nn
import numpy as np
from abc import abstractmethod
#from model.dynamic_model import BaseModel

# base model
class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

# dense model
class DenseModel(nn.Module):
    def __init__(self, input_dim=1, hidden_layers=2, neurons_per_layer=10, hidden_activation='relu', output_activation='linear'):
        super().__init__()

        # activation functions
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity()
        }

        layers = []
        prev_dim = input_dim

        # add hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, neurons_per_layer))
            layers.append(activations[hidden_activation])
            prev_dim = neurons_per_layer

        # output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(activations[output_activation])

        self.model = nn.Sequential(*layers)

    # forward pass
    def forward(self, x):
        return self.model(x)



# %%