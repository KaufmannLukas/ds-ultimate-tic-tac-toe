
from agents.agent import Agent
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


def test_ppo(ppo_agent: Agent, num_of_games=100):

    logger.info("Main started...")

    random_agent = Random()
    env = UltimateTicTacToeEnv(random_agent)


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

            ppo_color = game.white.color
            winner = game.winner
            ppo_wins = int(ppo_color == winner)
            ppo_loose = int(ppo_wins == 0 and winner is not None)
            draw = int(winner is None)

            winner_table.append(
                [i, 
                 ppo_color, 
                 winner, 
                 ppo_wins,
                 ppo_loose,
                 draw])
        else:
            winner_table.append(
                [i, game.black.color, game.winner])

    

    # Create CSV file with winning distribution
    
    #winner_table = []


    logger.info("ppo vs. random main stopped")
    return winner_table

def winner_table_to_dataframe(winner_table):
    winner_dataframe = pd.DataFrame(
        winner_table, columns=["game_nr", "ppo_color", "winner", "ppo_wins", "ppo_loose", "draw"])
    
    return winner_dataframe


if __name__ == "__main__":
    ppo_agent = PPO()
    # Load model values saved so far 
    ppo_agent.load("./data/ppo", "ppo_v4_2")

    winner_table = test_ppo(num_of_games=100,
         ppo_agent=ppo_agent)
    
    winner_df = winner_table_to_dataframe(winner_table)

    winner_df.to_csv(f"data/random_vs_ppo_v4_2.csv")