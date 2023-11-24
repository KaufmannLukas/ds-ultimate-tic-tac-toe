from agents.ppo import PPO
from agents.random_agnt import  Random
from agents.human import Agent
from gym_envs.uttt_env import UltimateTicTacToeEnv
from environments.game import Game
from agents.human import Human
from ppo_random import test_ppo, winner_table_to_dataframe

import pandas as pd

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename="PPO_test_main.log",
                    filemode='w+'
                    )


logger = logging.getLogger(__name__)

#from agents.network import FeedForwardNN

def train(model_name, model_path,
          ppo_hyperparameters,
          total_timesteps=1000,
          num_generations=1,
          save_generations=False,
          test_generations=True,
          num_of_test_games = 1000,
          ):

    logger.info("Start training...")

    # Agent = Agent / Random ?
    # TODO: load a model first 
    logger.info("Init PPO ...")
    ppo = PPO(name=model_name, path=model_path, hyperparameters=ppo_hyperparameters)

    logger.info("Init env ...")
    env = UltimateTicTacToeEnv(opponent=None, opponent_starts=False)

    logger.info("Start training ...")
    for i in range(num_generations):
        logger.info(f"Start generation {num_generations} ...")
        ppo.learn(env=env, total_timesteps=total_timesteps)
        if save_generations:
            ppo.save(name=model_name + f"_g{num_generations}", path=model_path)
        if test_generations:
            winner_table = test_ppo(ppo, num_of_games=num_of_test_games)
            winner_df = winner_table_to_dataframe(winner_table=winner_table)
            win_count = winner_df["ppo_wins"].sum()
            loose_count = winner_df["ppo_loose"]
            draw_count = winner_df["draw"]

            logger.info(f"Played {num_of_test_games} games against Random:")
            logger.info(f"winns: {win_count}\t looses: {loose_count}\t draws: {draw_count}")
        


    logger.info("End training...")


if __name__ == "__main__":
    total_timesteps = 1_000_000
    num_generations = 5
    model = "ppo_v_ppo_v1_0"
    path = "./data/ppo"


    ppo_hyperparameters = {
        "timesteps_per_batch": 1000,        # timesteps per batch
        "max_timesteps_per_episode": 1000,  # timesteps per episode
        "gamma": 0.95,                      # Discount factor to be applied when calculating Rewards-To-Go
        "n_updates_per_iteration": 10,      # Number of times to update actor/critic per iteration
        "clip": 0.2,                        # As recommended by the paper
        "lr": 0.0005,                        # Learning rate of actor optimizer
        "save_freq": 10,                  # How often we save in number of iterations 
    }


    train(total_timesteps=total_timesteps,
          ppo_hyperparameters=ppo_hyperparameters,
          model_name=model,
          model_path=path,
          num_generations=num_generations,
          )
    #play_ppo()