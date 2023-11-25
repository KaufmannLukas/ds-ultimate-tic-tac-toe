from agents.ppo import PPO
from agents.random_agnt import  Random
from agents.human import Agent
from gym_envs.uttt_env import UltimateTicTacToeEnv
from environments.game import Game
from agents.human import Human
from agents.mcts import MCTS
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
          load_name=None,
          load_path=None,
          total_timesteps=1000,
          num_generations=1,
          save_generations=False,
          test_generations=True,
          num_of_test_games = 1000,
          opponent = None,
          reward_config = None
          ):

    logger.info("Start training...")

    # Agent = Agent / Random ?
    # TODO: load a model first 
    logger.info("Init PPO ...")
    ppo = PPO(name=model_name,
              path=model_path,
              load_name=load_name,
              load_path=load_path,
              hyperparameters=ppo_hyperparameters,
              )

    logger.info("Init env ...")


    logger.info("Start training ...")
    for i in range(num_generations):
        env = UltimateTicTacToeEnv(opponent=opponent, opponent_starts=bool(i%2 == 0), reward_config=reward_config)
        logger.info(f"Start generation {i} ...")
        print(f"Start generation {i} ...")
        ppo.learn(env=env, total_timesteps=total_timesteps)
        if save_generations:
            ppo.save(name=model_name + f"_g{i}", path=model_path)
        if test_generations:
            logger.info("Start test vs random")
            print("Start test vs random")
            winner_table = test_ppo(ppo, num_of_games=num_of_test_games)
            winner_df = winner_table_to_dataframe(winner_table=winner_table)
            winner_df.to_csv("winner_table_ppo_v_ppo.csv")
            win_count = winner_df["ppo_wins"].sum()
            loose_count = winner_df["ppo_loose"].sum()
            draw_count = winner_df["draw"].sum()

            logger.info(f"Played {num_of_test_games} games against Random:")
            logger.info(f"winns: {win_count}\t looses: {loose_count}\t draws: {draw_count}")
            print(f"Played {num_of_test_games} games against Random:")
            print(f"winns: {win_count}\t looses: {loose_count}\t draws: {draw_count}")
        


    logger.info("End training...")


if __name__ == "__main__":
    total_timesteps = 500_000
    num_generations = 100

    model_name = "ppo_v_ppo_v1_8_hubraum"
    model_path = "./data/ppo/ppo_vs_ppo"

    # load_name = None
    # load_path = None
    load_name = "ppo_v_ppo_v1_7"
    load_path = "./data/ppo/ppo_vs_ppo"


    ppo_hyperparameters = {
        "timesteps_per_batch": 5000,        # timesteps per batch
        "max_timesteps_per_episode": 1000,  # timesteps per episode
        "gamma": 0.95,                      # Discount factor to be applied when calculating Rewards-To-Go
        "n_updates_per_iteration": 10,      # Number of times to update actor/critic per iteration
        "clip": 0.2,                        # As recommended by the paper
        "lr": 0.005,                        # Learning rate of actor optimizer
        "save_freq": 10,                    # How often we save in number of iterations 
    }

    reward_config = {
        "global_win_factor": 100,
        "global_draw_factor": 20,

        "local_win_factor": 5,
        "local_draw_factor": 2,

        "legal_move_factor": 0.1,
        # WAIT!!!!! READ BELOW
        "illegal_move_factor": -3,         # !!!!! CHANGE THE INVALID MOVE COUNT TO THE NUMBER !!!!!
        # dont forget to change the other number you IDIOTS!!!!!!
    }

    # SET STARTING OPPONENT HERE
    op_model = "ppo_v_ppo_v1_7"
    op_path = "./data/ppo/ppo_vs_ppo"
    #opponent = PPO(name=op_model, path=op_path, hyperparameters=ppo_hyperparameters)
    opponent = Random()

    train(total_timesteps=total_timesteps,
          ppo_hyperparameters=ppo_hyperparameters,
          model_name=model_name,
          model_path=model_path,
          load_name=load_name,
          load_path=load_path,
          num_generations=num_generations,
          save_generations=True,
          num_of_test_games = 100,
          opponent = opponent,
          reward_config = reward_config,
          )
    
