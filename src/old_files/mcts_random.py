from datetime import datetime
from environments.game import Game
from agents.mcts import MCTS, count_nodes, count_leaves
from agents.random import Random
import logging
import pickle

import pandas as pd
from tqdm import tqdm


# Format the date and time as a string with seconds precision and no spaces
formatted_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=f"logs/mcts_random_{formatted_date_time}.log",
                    filemode='w'
                    )


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logger.info("Main started...")

    num_iterations_list = [
        # 10,
        20,
        # 50,
        # 100,
        # 500,
        # 1000,
        # 2000,
        # 5000
    ]

    # logger.info("load mcts memory")
    # with open("data/mcts_ltmm.pkl", 'rb') as file:
    #     memory = pickle.load(file)

    num_of_games = 10
    random_agent = Random()

    winner_table = []

    for num_iterations in num_iterations_list:
        print(f"num_iterations: {num_iterations}")
        for i in tqdm(range(num_of_games)):
            mcts_agent = MCTS(
                memory_path="data/mcts_fluid_ltmm.pkl", update_memory=True)

            logger.info("Start new Game")
            game = Game()
            counter = 0

            # check which player's turn it is
            while not game.done:
                if (counter+i) % 2 == 0:
                    next_move = mcts_agent.play(
                        game, num_iterations=num_iterations)
                else:
                    next_move = random_agent.play(game)
                game.play(*next_move)
                counter += 1

            logger.info(f"node count: {count_nodes(mcts_agent.memory)}")
            logger.info(f"leave count: {count_leaves(mcts_agent.memory)}")
            mcts_agent.save_memory()
            logger.info("Game done")

            '''save results to csv-file with winner-information / statistics for model evaluation'''
            # if i % 2 == 0:
            #     # player_x = mcts_agent
            #     # player_o = random_agent
            #     winner_table.append(
            #         [i, num_iterations, game.white.color, game.winner])
            # else:
            #     winner_table.append(
            #         [i, num_iterations, game.black.color, game.winner])

        # winner_dataframe = pd.DataFrame(
        #     winner_table, columns=["game_nr", "num_iter", "mcts_color", "winner"])
        # winner_dataframe.to_csv(f"data/random_vs_mcts_{num_iterations}_memory_1.csv")
        # winner_table = []

        logger.info("mcts vs. random main stopped")
