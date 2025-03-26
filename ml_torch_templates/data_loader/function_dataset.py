# %%
# 1. implement custom dataset

import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

# simple functions dataset
class SimpleFunctionsDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.x = torch.rand(n_samples) * (2 * math.pi)  
        self.function = function
        
        # y based on function type
        noise = (torch.rand(n_samples) * 2 - 1)  
        if function == 'linear':
            self.y = 1.5 * self.x + 0.3 + noise
        elif function == 'quadratic':
            self.y = 2 * self.x**2 + 0.5 * self.x + 0.3 + noise
        elif function == 'harmonic':
            self.y = 0.5 * self.x**2 + 5 * torch.sin(self.x) + 3 * torch.cos(3 * self.x) + 2 + noise
        else:
            raise ValueError("Function type must be 'linear', 'quadratic', or 'harmonic'.")
        
        # normalize y 
        self.y = (self.y - self.y.mean()) / self.y.std()
    
    # Return length of dataset
    def __len__(self):
        return len(self.x)
    
    # x and y values for each sample
    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx]  # x should be a tensor of shape [1]

# data loader
class CustomDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        dataset = SimpleFunctionsDataset(n_samples=n_samples, function=function)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# %%
