from datetime import datetime
from agents.agent import Agent
from environments.game import Game
from agents.ppo import PPO
from agents.random import Random
from environments.uttt_env import UltimateTicTacToeEnv
from agents.human import Human

import logging


import pandas as pd
from tqdm import tqdm


# Format the date and time as a string with seconds precision and no spaces
formatted_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=f"logs/test_ppo_{formatted_date_time}.log",
                    filemode='w'
                    )


logger = logging.getLogger(__name__)


def test_ppo(ppo_agent: Agent, opponent: Agent, num_of_games=100, print_stuff=False):
    """
    Test the PPO agent against an opponent in a specified number of games.

    Parameters:
    - ppo_agent (Agent): The PPO agent being tested.
    - opponent (Agent): The opponent agent.
    - num_of_games (int): Number of games to play.
    - print_stuff (bool): Whether to print the game progress.

    Returns:
    - list: List containing game results.
    """

    logger.info("Main started...")

    winner_table = []

    for i in tqdm(range(num_of_games)):
        
        game = Game()
        counter = 0

        # check which player's turn it is
        while not game.done:
            if print_stuff:
                print(game)
            if (counter+i) % 2 == 0:
                next_move = ppo_agent.play(game)
            else:
                next_move = opponent.play(game)
            if print_stuff:
                print(next_move)
            if not game.check_valid_move(*next_move):
                if print_stuff:
                    print("Invalid Move, restart the Game...")
                game = Game()
                continue
            game.play(*next_move)
            counter += 1
            if game.check_draw()[0]:
                break

        if i % 2 == 0:
            ppo_color = game.white.color
        else:
            ppo_color = game.black.color

        winner = game.winner
        ppo_wins = int(ppo_color == winner)
        ppo_loose = int(ppo_wins == 0 and winner is not None)
        draw = int(game.global_draw)  

        winner_table.append(
            [i, 
                ppo_color, 
                winner, 
                ppo_wins,
                ppo_loose,
                draw])

    logger.info("ppo vs random main stopped")

    return winner_table


def winner_table_to_dataframe(winner_table):
    """
    Convert the winner table list to a pandas DataFrame.

    Parameters:
    - winner_table (list): List containing game results.

    Returns:
    - pd.DataFrame: Pandas DataFrame with the game results.
    """
    winner_dataframe = pd.DataFrame(
        winner_table, columns=["game_nr", "ppo_color", "winner", "ppo_wins", "ppo_loose", "draw"])
    
    return winner_dataframe


if __name__ == "__main__":
    # Load model values saved so far 
    model = "ppo_v1"
    path = "./data/models/ppo"
    ppo_agent = PPO(name=model, path=path)

    num_of_games = 100

    random_agent = Random()
    human_agent = Human()

    winner_table = test_ppo(
        num_of_games=num_of_games,
         ppo_agent=ppo_agent,
         opponent=random_agent,
         print_stuff=False)
    
    winner_df = winner_table_to_dataframe(winner_table)

    winner_df.to_csv(f"data/local/test_ppo_vs_random.csv")

    win_count = winner_df["ppo_wins"].sum()
    loose_count = winner_df["ppo_loose"].sum()
    draw_count = winner_df["draw"].sum()

    print(f"Played {num_of_games} games against Random:")
    print(f"wins: {win_count}\t losses: {loose_count}\t draws: {draw_count}")