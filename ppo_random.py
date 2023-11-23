
from environments.game import Game
from agents.ppo import PPO
from agents.random_agnt import Random
from gym_envs.uttt_env import UltimateTicTacToeEnv

import logging

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename="log_ppo_random.log",
                    filemode='w'
                    )


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logger.info("Main started...")

    num_of_games = 1000
    random_agent = Random()
    env = UltimateTicTacToeEnv(random_agent)
    ppo_agent = PPO(env, show_probs=True)
    # Load model values saved so far 
    ppo_agent.load("./data/ppo", "ppo_v5_1")

    winner_table = []



    for i in tqdm(range(num_of_games)):
        
        logger.info("Start new Game")
        game = Game()
        counter = 0

        # check which player's turn it is
        while not game.done:
            print(game)
            if (counter+i) % 2 == 0:
                next_move = ppo_agent.play(game)
            else:
                next_move = random_agent.play(game)
            print(next_move)
            if not game.check_valid_move(*next_move):
                print("Invalid Move, restart the Game...")
                game = Game()
                break
            game.play(*next_move)
            counter += 1

        logger.info("Game done")

        if i % 2 == 0:
            # player_x = mcts_agent
            # player_o = random_agent
            winner_table.append(
                [i, game.white.color, game.winner])
        else:
            winner_table.append(
                [i, game.black.color, game.winner])

    winner_dataframe = pd.DataFrame(
        winner_table, columns=["game_nr", "ppo_color", "winner"])

    # Create CSV file with winning distribution
    winner_dataframe.to_csv(f"data/random_vs_ppo_v5_1.csv")

    wins = winner_dataframe[winner_dataframe["winner"] == winner_dataframe["ppo_color"]].value_counts()
    draws = winner_dataframe["winner"].isna().sum()
    losses = winner_dataframe[winner_dataframe["winner"] != winner_dataframe["ppo_color"]].value_counts()

    print(f"The ppo_v4_ppo has won {wins.shape[0]} out of {num_of_games} games against Baseline Model.")
    print(f"The ppo_v4_ppo has tied {draws} out of {num_of_games} games against Baseline Model.")
    print(f"The ppo_v4_ppo has lost {losses.shape[0]} out of {num_of_games} games against Baseline Model.")

    #winner_table = []


    logger.info("ppo vs. random main stopped")
