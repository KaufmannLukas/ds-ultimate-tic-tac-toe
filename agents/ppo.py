# TODO: comment all functions

"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time
import logging

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam
from tqdm import tqdm

from agents.agent import Agent
# from agents.network_v2 import FeedForwardNN_Actor, FeedForwardNN_Critic
from agents.network import FeedForwardNN_Actor, FeedForwardNN_Critic
from gym_envs.uttt_env import game2tensor

logger = logging.getLogger(__name__)



class PPO(Agent):
    """
        This is the PPO class we will use as our model in main.py
    """

    def __init__(self, env, name=None, path=None, show_probs=False, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """

        logger.info("Init PPO agent...")

        assert(type(env.observation_space) == gym.spaces.box.Box)
        assert(type(env.action_space) == gym.spaces.box.Box)
        
        self._init_hyperparameters()
        self.name = name
        self.path = path

        self.show_probs = show_probs


        # This logger will help us with printing out summaries of each iteration
        self.logger_dict = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

        self.reward_history = []

        # Extract environment information
        # TODO: check if shape[0] makes sense for us
        self.env = env
        # For a 4x9x9 observation space, this should be 324
        self.obs_dim = np.prod(env.observation_space.shape)
        self.act_dim = 81 # env.action_space.shape

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN_Actor(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN_Critic(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

    # TODO: adjust values later

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 5000        # timesteps per batch
        self.max_timesteps_per_episode = 1000  # timesteps per episode
        self.gamma = 0.95                      # Discount factor to be applied when calculating Rewards-To-Go
        self.n_updates_per_iteration = 10      # Number of times to update actor/critic per iteration
        self.clip = 0.2                        # As recommended by the paper
        self.lr = 0.0005                        # Learning rate of actor optimizer
        self.save_freq = 10                  # How often we save in number of iterations 

    def learn(self, total_timesteps):

        logger.info(f"start learn with {total_timesteps} total timesteps...")
        
        pbar = tqdm(total=total_timesteps)

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far


        while t_so_far < total_timesteps:              # ALG STEP 2

            # Update the progress bar to reflect the current timestep
            pbar.n = t_so_far
            pbar.refresh()  # Refresh the progress bar to show the updated value

            # Increment t_so_far somewhere below
            # TODO: for loop makes more sense?
            # ALG STEP 3
            # ollecting our batch simulations
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger_dict['t_so_far'] = t_so_far
            self.logger_dict['i_so_far'] = i_so_far

            
            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            # TODO: check if usefull for us or not
            '''One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            isn't theoretically necessary, but in practice it decreases the variance of 
            our advantages and makes convergence much more stable and faster. I added this because
            solving some environments was too unstable without it.'''
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True) # TODO: maybe with retain_graph=True (see original repo)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger_dict['actor_losses'].append(actor_loss.detach())

            if i_so_far % self.save_freq == 0:
                
                torch.save(self.actor.state_dict(), self.path+"/"+self.name+"_actor.pth")
                torch.save(self.critic.state_dict(), self.path+"/"+self.name+"_critic.pth")
                logger.info("saved files")
                self._log_summary()
                print("model saved")

        # Print a summary of our training so far
        self._log_summary()

        # Save our model if it's time
        
        pbar.close()


    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch
        batch_rew_count_dicts = [] # list of dictionaries for collected rewards 

        # Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
        ep_rews = []   

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            # count of rewards dic => in order to track invalid moves
            rew_count_dict = {}
            obs, _ = self.env.reset()
            done = False

            # TODO: make a while loop instead because max_timesteps_per_episode will never reached
            for ep_t in range(self.max_timesteps_per_episode):

                #self.env.render()
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action) # two _ variables because gym changed it in newer version

                rew_count_dict[rew] = rew_count_dict.get(rew, 0) + 1



                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
            batch_rew_count_dicts.append(rew_count_dict)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        if len(batch_acts) > 0:
            batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        else:
            # Handle the case where batch_acts is empty or not properly formatted
            raise ValueError("batch_acts is empty or not in the correct format")

        if not isinstance(batch_acts, torch.Tensor):
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        else:
            batch_acts = batch_acts.clone().detach()
        # Reshape data as tensors in the shape specified before returning
        #batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        #batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths and reward count (for invalid_move count) in this batch.
        self.logger_dict['batch_rews'] = batch_rews
        self.logger_dict['batch_lens'] = batch_lens
        self.logger_dict['batch_rew_count_dicts'] = batch_rew_count_dicts


        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, obs, mode="learn"):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        # TODO: check later if mean makes sense in discrete, not continuous action space.

        # Convert obs to a PyTorch tensor if it's not already one
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)

        flat_obs = obs.view(-1)

        probs = self.actor(flat_obs)


        # If you wanna print the probabilities, yes, if not, comment out
        if mode == "play":
            print("probs before masking: ")
            print(probs)

        '''putting probs for invalid moves to X manually'''
        #blocked_fields = obs[0:81]
        # probs[blocked_fields == 1] = torch.finfo(torch.float64).eps

        '''Normalisation of the probs'''
        #print("Probs before normalization:", probs)
        #probs[blocked_fields == 1] = 0
        #probs /= (probs.sum() + 1e-8)  # Adding a small epsilon to avoid division by zero
        #print("Probs after normalization:", probs)

        if mode == "play" and self.show_probs:
            print("probs: ")
            print(probs)

        # Create our Multivariate Normal Distribution
        #dist = MultivariateNormal(mean, self.cov_mat)
        m = Categorical(probs)
        # Sample an action from the distribution and get its log prob
        action = m.sample() # asuming this sample is NOT random ???

        # print("action: ", action)

        # next_state, reward = env.step(action)
        # loss = -m.log_prob(action) * reward
        # loss.backward()


        log_prob = m.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor.forward(batch_obs)
        #dist = MultivariateNormal(mean, self.cov_mat)
        dist = Categorical(mean)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs

        return V, log_probs


    def play(self, game):
        obs = game2tensor(game)
        action, _ = self.get_action(obs, mode="play")

    
        game_idx = action // 9
        field_idx = action % 9

        move = (game_idx, field_idx)

        return (game_idx, field_idx)


    def save(self, path, name):
        torch.save(self.actor.state_dict(), path+"/"+name+"_actor.pth")
        torch.save(self.critic.state_dict(), path+"/"+name+"_critic.pth")
        
    def load(self, path, name):
        self.actor.load_state_dict(torch.load(path + "/" + name + "_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "/" + name + "_critic.pth"))


    def _log_summary(self):
            """
                Print to stdout what we've logged so far in the most recent batch.

                Parameters:
                    None

                Return:
                    None
            """
            # Calculate logging values. I use a few python shortcuts to calculate each value
            # without explaining since it's not too important to PPO; feel free to look it over,
            # and if you have any questions you can email me (look at bottom of README)
            delta_t = self.logger_dict['delta_t']
            self.logger_dict['delta_t'] = time.time_ns()
            delta_t = (self.logger_dict['delta_t'] - delta_t) / 1e9
            delta_t = str(round(delta_t, 2))

            t_so_far = self.logger_dict['t_so_far']
            i_so_far = self.logger_dict['i_so_far']
            avg_ep_lens = np.mean(self.logger_dict['batch_lens'])
            avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger_dict['batch_rews']])
            avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger_dict['actor_losses']])

            rew_dict_list = self.logger_dict["batch_rew_count_dicts"]

            invalid_move_count = sum(rew_dict[-2] for rew_dict in rew_dict_list if -2 in rew_dict.keys())
            invalid_move_ratio = invalid_move_count / self.timesteps_per_batch
            
            # Round decimal places for more aesthetic logging messages
            avg_ep_lens = str(round(avg_ep_lens, 2))
            avg_ep_rews = str(round(avg_ep_rews, 2))
            avg_actor_loss = str(round(avg_actor_loss, 5))
            invalid_move_ratio = str(round(invalid_move_ratio, 5))


            # Print logging statements
            print(flush=True)
            print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
            print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
            print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            print(f"Average Loss: {avg_actor_loss}", flush=True)
            print(f"Timesteps So Far: {t_so_far}", flush=True)
            print(f"Iteration took: {delta_t} secs", flush=True)
            print(f"invalid move count: {invalid_move_count}", flush=True)
            print(f"invalid move ratio: {invalid_move_ratio}", flush=True)
            print(f"------------------------------------------------------", flush=True)
            print(flush=True)

            logger.info(f"-------------------- Iteration #{i_so_far} --------------------")
            logger.info(f"Average Episodic Length: {avg_ep_lens}")
            logger.info(f"Average Episodic Return: {avg_ep_rews}")
            logger.info(f"Average Loss: {avg_actor_loss}")
            logger.info(f"Timesteps So Far: {t_so_far}")
            logger.info(f"Iteration took: {delta_t} secs")
            logger.info(f"invalid move count: {invalid_move_count}")
            logger.info(f"invalid move ratio: {invalid_move_ratio}")
            logger.info(f"------------------------------------------------------" )

            # Reset batch-specific logging data
            self.logger_dict['batch_lens'] = []
            self.logger_dict['batch_rews'] = []
            self.logger_dict['actor_losses'] = []
            self.logger_dict['batch_rew_count_dicts']



