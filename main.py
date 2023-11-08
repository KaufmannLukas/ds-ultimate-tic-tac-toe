
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

    games_data_history = []
    games_data_wins = []

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
        wins = [
            env.game.white.wins.sum(),
            env.game.black.wins.sum(),
            env.game.winner,

        ]

        games_data_wins.append(wins)

        game_moves = combine_moves(
            env.game.white.history, env.game.black.history)
        games_data_history.append(game_moves)

    df_games = pd.DataFrame(games_data_history, columns=[
                            str(i) for i in range(81)])
    df_wins = pd.DataFrame(games_data_wins, columns=[
                           "white_locals", "black_locals", "winner"])

    end_time = time.time()
    print("end:", end_time)
    duration = end_time - start_time
    print("duration: ", duration)

    df_games.to_csv("data/history.csv")
    df_wins.to_csv("data/wins.csv")
    # move_history.to_csv("history.csv")

    print("done")

    # print(f"white rewards: {reward_history_white}", sum(reward_history_white))
    # print(f"black rewards: {reward_history_black}", sum(reward_history_black))
