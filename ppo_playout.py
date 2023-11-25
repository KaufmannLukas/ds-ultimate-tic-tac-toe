
from agents.agent import Agent
from environments.game import Game
from agents.ppo import PPO
from agents.random_agnt import Random
from gym_envs.uttt_env import UltimateTicTacToeEnv
from agents.human import Human

import logging

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename="log_ppo_random.log",
                    filemode='w'
                    )


logger = logging.getLogger(__name__)


def test_ppo(ppo_agent: Agent, opponent: Agent, num_of_games=100, print_stuff=False):

    logger.info("Main started...")

    
    env = UltimateTicTacToeEnv(opponent)


    winner_table = []



    for i in tqdm(range(num_of_games)):
        
        #logger.info("Start new Game")
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

        #logger.info("Game done")

        if i % 2 == 0:
            # player_x = ppo_agent
            # player_o = opponent
            ppo_color = game.white.color
        else:
            ppo_color = game.black.color

        winner = game.winner
        ppo_wins = int(ppo_color == winner)
        ppo_loose = int(ppo_wins == 0 and winner is not None)
        #draw = int(winner is None)   
        # draw = int(game.check_draw()[0] == True)
        draw = int(game.global_draw)  

        winner_table.append(
            [i, 
                ppo_color, 
                winner, 
                ppo_wins,
                ppo_loose,
                draw])


    logger.info("ppo vs. random main stopped")
    return winner_table

def winner_table_to_dataframe(winner_table):
    winner_dataframe = pd.DataFrame(
        winner_table, columns=["game_nr", "ppo_color", "winner", "ppo_wins", "ppo_loose", "draw"])
    
    return winner_dataframe


if __name__ == "__main__":
    # Load model values saved so far 

    model = "ppo_v_ppo_v1_2"
    path = "./data/ppo/ppo_vs_ppo"
    ppo_agent = PPO(name=model, path=path)
    #ppo_agent.load(path, model)

    num_of_games = 1

    random_agent = Random()
    human_agent = Human()

    winner_table = test_ppo(
        num_of_games=num_of_games,
         ppo_agent=ppo_agent,
         opponent=human_agent,
         print_stuff=True)
    
    winner_df = winner_table_to_dataframe(winner_table)

    #winner_df.to_csv(f"data/random_vs_ppo_v4_2.csv")

    win_count = winner_df["ppo_wins"].sum()
    loose_count = winner_df["ppo_loose"].sum()
    draw_count = winner_df["draw"].sum()

    print(f"Played {num_of_games} games against Random:")
    print(f"winns: {win_count}\t looses: {loose_count}\t draws: {draw_count}")