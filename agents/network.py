import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNetwork(nn.Module):

    # get input and output dimensions from 'uttt_env.py'
    # (currently: 4x9x9, TODO: check if has to change)
    def __init__(self, in_dim, out_dim):
        super(ActorCriticNetwork, self).__init__()

        hidden_dim = 64

        #in_dim = 324  # 4 * 9 * 9
        #out_dim = 81  # 9 * 9

        # TODO: adjust layers/architecture later (64, etc.)
        self.shared_layer1 = nn.Linear(in_dim, 64)
        self.shared_layer2 = nn.Linear(64, 64)

        #Actor specific layers
        self.actor_output_layer = nn.Linear(hidden_dim, out_dim)
        self.actor_output_activation = nn.Softmax(dim=-1)

        #Critic specific layers
        self.critic_output_layer = nn.Linear(hidden_dim, 1)
        self.critic_output_activation = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for layer in [self.shared_layer1, self.shared_layer2, self.actor_output_layer, self.critic_output_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, obs):
        # Convert observation to tensor if it's a numpy array
        # if isinstance(obs, np.ndarray):
        #     obs_new = torch.tensor(obs, dtype=torch.float)
        # else:
        #     obs_new = obs
        
        # # TODO: maybe change activation function for improvement?
        # activation1 = F.relu(self.layer1(obs_new))
        # activation2 = F.relu(self.layer2(activation1))
        # output = self.layer3(activation2)
        # return output

        if isinstance(obs, np.ndarray):
            obs_new = torch.tensor(obs, dtype=torch.float)
        else:
            obs_new = obs
            
        shared_activation1 = F.relu(self.shared_layer1(obs_new))
        shared_activation2 = F.relu(self.shared_layer2(shared_activation1))
        
        # Actor network
        actor_output = self.actor_output_activation(self.actor_output_layer(shared_activation2))

        # Critic network
        critic_output = self.critic_output_activation(self.critic_output_layer(shared_activation2))

        return actor_output, critic_output