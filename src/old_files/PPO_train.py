import datetime
from agents.ppo import PPO
from agents.random import  Random
from agents.human import Agent
from environments.uttt_env import UltimateTicTacToeEnv
from environments.game import Game
from agents.human import Human

import pandas as pd

import logging

# Format the date and time as a string with seconds precision and no spaces
formatted_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=f"logs/PPO_train_{formatted_date_time}.log",
                    filemode='w'
                    )


logger = logging.getLogger(__name__)

#from agents.network import FeedForwardNN

def train_ppo(total_timesteps):

    logger.info("Start training...")

    # Agent = Agent / Random ?
    # TODO: load a model first 

    random_opponent = Random()

    env = UltimateTicTacToeEnv(opponent=random_opponent, opponent_starts=False)

    print(env.opponent)
    # Create a model for PPO.
    model = PPO(env=env, name="ppo_v4_5_test", path="./data/ppo")
    #if model_path and model_name:
    # model.load(model_path, model_name)
    # model = PPO(policy_class=FeedForwardNN, env=env)
    #model = PPO(env)

    #for i in range(1):
    model.learn(total_timesteps=total_timesteps)
    model.save(name="ppo_v4_5_test", path="./data/ppo")

    # Saving the reward history
    # Comment this out if you're training with large numbers!
   #pd.DataFrame(env.full_reward_history).to_csv("./full_reward_history.csv")

    logger.info("End training...")


def play_ppo():
    env = UltimateTicTacToeEnv(opponent=None, opponent_starts=False)
    model = PPO(name="ppo_v1", path="./data/models/ppo")
    #model.load("./data/ppo", "ppo_vs_")
    game = Game()
    #implement the next two lines for using a memory_prone agent (like mcts_agent_01)
    # with open("data/mcts_ltmm_02.pkl", mode="rb") as file:
    #     memory = pickle.load(file)
    #change the agent to play against here:
    computer_agent = model
    human_agent = Human()

    # TODO: improve interface (print local/global wins, etc.)
    # TODO: implement game loop (replay)
    counter = 0
    while not game.done:
        print("-"*31)
        print(game)
        print("-"*31)
        if counter % 2 != 0:
            next_move = human_agent.play(game)
        else:
            next_move = computer_agent.play(game)
            mc = 0
            while True:
                if not game.check_valid_move(*next_move) and mc < 10:
                    print(next_move)
                    next_move = computer_agent.play(game)
                    mc += 1
                else:
                    break
        print(f"last move: ({next_move[0]+1}, {next_move[1]+1})")
        result = game.play(*next_move)
        print(result)
        counter += 1

    print(game)


if __name__ == "__main__":
    total_timesteps = 1_000_000
    #train_ppo(total_timesteps)
    play_ppo()