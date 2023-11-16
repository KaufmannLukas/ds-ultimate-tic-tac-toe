from agents.ppo import PPO
from agents.random_agnt import  Random
from agents.human import Agent
from gym_envs.uttt_env import UltimateTicTacToeEnv
from agents.network import FeedForwardNN

if __name__ == "__main__":

    # Agent = Agent / Random ?

    env = UltimateTicTacToeEnv(Agent)


    print(env.opponent)
    # Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env)
    #model = PPO(env)
    model.learn(10000)
