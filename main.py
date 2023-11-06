
from gym_envs.uttt_env import UltimateTicTacToeEnv, random_policy
import pandas as pd
import numpy as np
from tqdm import tqdm

import time


def combine_moves(white_moves, black_moves):
    # Interleave the moves, assuming white starts first
    moves = [None]*81  # Initialize a list with 81 moves
    moves[:len(white_moves)*2:2] = white_moves
    moves[1:len(black_moves)*2:2] = black_moves
    return moves



if __name__ == "__main__":
    
    env = UltimateTicTacToeEnv()

    reward_history_white = []
    reward_history_black = []
    counter = 0

    move_history = pd.DataFrame(columns=[i for i in range(81)])
    games_data = []

    start_time = time.time()
    print("start: ", start_time)


    for i in tqdm(range(1000)):

        done = False
        state = env.reset()

        while not done:
            action = random_policy(state)
            state, reward, done, _ = env.step(action)

            if counter % 2:
                reward_history_black.append(reward)
            else:
                reward_history_white.append(reward)

            # env.render()

            # print("")
            # print("-"*31)
            # print("")
            counter = counter + 1
        

        game_moves = combine_moves(env.game.white.history, env.game.black.history)
        games_data.append(game_moves)
        
    df_games = pd.DataFrame(games_data, columns=[str(i) for i in range(81)])


    end_time = time.time()
    print("end:", end_time)
    duration = end_time - start_time
    print("duration: ", duration)

    df_games.to_csv("history.csv")
    #move_history.to_csv("history.csv")



    print("done")


        # print(f"white rewards: {reward_history_white}", sum(reward_history_white))
        # print(f"black rewards: {reward_history_black}", sum(reward_history_black))
    