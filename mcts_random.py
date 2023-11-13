
from environments.game import Game
from agents.mcts import MCTS
from agents.random import Random
import logging
import pickle

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename="log_mcts_random.log",
                    filemode='w+'
                    )


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logger.log(logging.INFO, "Main started...")

    num_iterations_list = [
        10,
        20,
        # 50,
        # 100,
        # 500,
        # 1000,
        # 2000,
        # 5000
    ]

    logger.info("load mcts memory")
    with open("data/mcts_ltmm.pkl", 'rb') as file:
        memory = pickle.load(file)

    num_of_games = 10
    random_agent = Random()

    winner_table = []

    for num_iterations in num_iterations_list:
        print(f"num_iterations: {num_iterations}")
        for i in tqdm(range(num_of_games)):
            mcts_agent = MCTS(memory=memory)

            logger.info("Stat new Game")
            game = Game()
            counter = 0

            while not game.done:
                if (counter+i) % 2 == 0:
                    next_move = mcts_agent.play(
                        game, num_iterations=num_iterations)
                else:
                    next_move = random_agent.play(game)
                game.play(*next_move)
                counter += 1
            logger.info("Game done")
            if i % 2 == 0:
                # player_x = mcts_agent
                # player_o = random_agent
                winner_table.append(
                    [i, num_iterations, game.white.color, game.winner])
            else:
                winner_table.append(
                    [i, num_iterations, game.black.color, game.winner])

        # winner_dataframe = pd.DataFrame(
        #     winner_table, columns=["game_nr", "num_iter", "mcts_color", "winner"])
        # winner_dataframe.to_csv(f"data/random_vs_mcts_{num_iterations}.csv")
        # winner_table = []

        logger.info("mcts main stopped")