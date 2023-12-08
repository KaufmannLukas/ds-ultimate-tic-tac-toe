from datetime import datetime
from agents.ppo import PPO
from agents.random import  Random
from environments.uttt_env import UltimateTicTacToeEnv
from agents.mcts import MCTS
from src.test_ppo import test_ppo, winner_table_to_dataframe

import logging

# Format the date and time as a string with seconds precision and no spaces
formatted_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=f"logs/train_ppo_{formatted_date_time}.log",
                    filemode='w'
                    )

logger = logging.getLogger(__name__)


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
    
    """
    Train a Proximal Policy Optimization (PPO) agent.

    Parameters:
    - model_name (str): Name of the PPO agent.
    - model_path (str): Path to save the PPO model.
    - ppo_hyperparameters (dict): Dictionary containing PPO hyperparameters.
    - load_name (str): Name of the model to load (optional).
    - load_path (str): Path from which to load the model (optional).
    - total_timesteps (int): Total number of training timesteps.
    - num_generations (int): Number of training generations.
    - save_generations (bool): Whether to save models after each generation.
    - test_generations (bool): Whether to test the agent against a random opponent after each generation.
    - num_of_test_games (int): Number of test games against a random opponent.
    - opponent (Agent): Opponent agent for testing.
    - reward_config (dict): Configuration for reward factors.
    """

    logger.info("Start training...")
    logger.info("Init PPO ...")

    ppo = PPO(name=model_name,
              path=model_path,
              load_name=load_name,
              load_path=load_path,
              hyperparameters=ppo_hyperparameters,
              )

    logger.info("Init env ...")

    test_opponent = Random()


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
            winner_table = test_ppo(ppo, opponent=test_opponent, num_of_games=num_of_test_games)
            winner_df = winner_table_to_dataframe(winner_table=winner_table)
            winner_df.to_csv("data/local/train_ppo_winner_table_01.csv")
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
    num_generations = 5

    model_name = "ppo_v2"
    model_path = "./data/models/ppo"

    # load_name = None
    # load_path = None
    load_name = "ppo_v1"
    load_path = "./data/models/ppo"


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

        "legal_move_factor": 1,
        # NOTE: !!!!
        # change the invalid_move_count "here",
        # if you change the following illegal_move_factor!!!!!
        "illegal_move_factor": -0.1,
        # dont forget to change the other number you IDIOTS!!!!!!
        # TODO: automate this!
    }

    # SET STARTING OPPONENT HERE
    # NOTE: if you want to train ppo vs. ppo, uncomment the following lines:
    # op_model = "ppo_v1"
    # op_path = "./data/models/ppo"
    # opponent = PPO(name=op_model, path=op_path, hyperparameters=ppo_hyperparameters)
    # NOTE: comment out the next line if you train ppo vs. ppo
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
    
