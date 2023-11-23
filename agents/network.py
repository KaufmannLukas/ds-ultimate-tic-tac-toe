import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN_Actor(nn.Module):

    # get input and output dimensions from 'uttt_env.py'
    # (currently: 4x9x9, TODO: check if has to change)
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN_Actor, self).__init__()

        #in_dim = 324  # 4 * 9 * 9
        #out_dim = 81  # 9 * 9

        # TODO: adjust layers later (64)
        self.layer1 = nn.Linear(in_dim, 324)
        self.bn1 = nn.BatchNorm1d(324)
        self.layer2 = nn.Linear(324, 324)
        self.bn2 = nn.BatchNorm1d(324)
        self.layer3 = nn.Linear(324, 324)
        self.bn3 = nn.BatchNorm1d(324)
        # self.layer4 = nn.Linear(324, 324)
        # self.layer5 = nn.Linear(324, 324)
        self.layer6 = nn.Linear(324, out_dim)


        # self.layer1 = nn.Linear(in_dim, 128)
        # self.bn1 = nn.BatchNorm1d(128)  # Batch normalization layer
        # self.layer2 = nn.Linear(128, 128)
        # self.bn2 = nn.BatchNorm1d(128)  # Batch normalization layer
        # self.layer3 = nn.Linear(128, out_dim)

        # Consider adding batch normalization layers, especially if you are deepening your network.
        # Batch normalization can improve the stability and speed of training.


    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        # TODO: maybe change activation function for improvement?
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        # activation4 = F.relu(self.layer4(activation3))
        # activation5 = F.relu(self.layer5(activation4))
        output = F.softmax(self.layer6(activation3), dim=-1)
        return output


class FeedForwardNN_Critic(nn.Module):

    # get input and output dimensions from 'uttt_env.py'
    # (currently: 4x9x9, TODO: check if has to change)
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN_Critic, self).__init__()

        #in_dim = 324  # 4 * 9 * 9
        #out_dim = 81  # 9 * 9

        # TODO: adjust layers later (324)
        self.layer1 = nn.Linear(in_dim, 324)
        self.bn1 = nn.BatchNorm1d(324)
        self.layer2 = nn.Linear(324, 324)
        self.bn2 = nn.BatchNorm1d(324)
        self.layer3 = nn.Linear(324, 324)
        self.bn3 = nn.BatchNorm1d(324)
        # self.layer4 = nn.Linear(324, 324)
        # self.layer5 = nn.Linear(324, 324)
        self.layer6 = nn.Linear(324, out_dim)


    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        obs = obs.view(obs.size(0), -1)  # Reshape obs to be [batch_size, feature_size]

        
        # TODO: maybe change activation function for improvement?

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        # activation4 = F.relu(self.layer4(activation3))
        # activation5 = F.relu(self.layer5(activation4))
        output = self.layer6(activation3)

        # activation1 = F.relu(self.layer1(obs))
        # activation2 = F.relu(self.layer2(activation1))
        # #output = nn.Identity(self.layer3(activation2))
        # output = self.layer3(activation2)
        return output
    