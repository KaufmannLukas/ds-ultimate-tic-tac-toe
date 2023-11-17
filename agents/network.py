import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):

    # get input and output dimensions from 'uttt_env.py'
    # (currently: 4x9x9, TODO: check if has to change)
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        in_dim = 324  # 4 * 9 * 9
        out_dim = 81  # 9 * 9

        # TODO: adjust layers later (64)
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)


    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        # TODO: maybe change activation function for improvement?
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

