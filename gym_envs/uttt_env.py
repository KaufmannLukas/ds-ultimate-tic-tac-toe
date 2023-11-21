from random import choice
from anyio import open_process
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib.pylab import get_state

from environments.game import Game
from agents.agent import Agent


class UltimateTicTacToeEnv(gym.Env):
    def __init__(self, opponent: Agent = None, opponent_starts=False):
        super(UltimateTicTacToeEnv, self).__init__()
        self.action_space = spaces.Box(
            low=0, high=1, shape=(9, 9), dtype=float)  # 9x9 board

        # 9x9 game boards for white, black, last_move, blocked_field
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(4, 9, 9), dtype=bool)
        self.game = Game()


        self.opponent = opponent
        self.opponent_starts = opponent_starts
    
        self.reset()

    def reset(self):
        self.game = Game()
        if self.opponent is not None and self.opponent_starts:
                opponent_move = self.opponent.play(self.game)
                self.game.play(*opponent_move)
        return game2tensor(self.game), {}
        

    def step(self, action: int):
        '''
        action: tuple like (game_idx, field_idx)
        '''

        
        reward = 0

        # action_list = [(i, p) for i, p in enumerate(action.tolist())]
        # action_list.sort(key=lambda x: x[1], reverse=True)
        # for action in action_list:
        #     move = Game.get_index_from_vector(action[0])
        #     if not self.game.check_valid_move(*move):
        #         reward -= 1
        #     else:
        #         break


        #action = move

        game_idx = action // 9
        field_idx = action % 9

        move = (game_idx, field_idx)

        if self.game.check_valid_move(*move):
            self.game.play(*move)

            if self.opponent is not None and not self.game.done:
                counter_action = self.opponent.play(self.game)
                self.game.play(*counter_action)
        else:
            reward -= 1



        new_state = game2tensor(self.game)
        done = self.game.done

        return new_state, reward, done, {}, {}


    def render(self):
        print(self.game)

def game2tensor(game: Game):

    wb = game.white.board
    bb = game.black.board
    bf = game.blocked_fields
    lm = np.zeros(shape=(9, 9), dtype=bool)
    if game.last_move:
        lm[*game.last_move] = True

    res = np.stack([wb, bb, bf, lm])
    return res.flatten()

'''
class UltimateTicTacToeEnv(gym.Env):
    def __init__(self):
        super(UltimateTicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(81)  # 9x9 board
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(4, 9, 9), dtype=bool)
        self.game = Game()

    def reset(self):
        # Initialize your game here and return the initial state
        self.game = Game()
        return self.get_state()

    def step(self, action):
        # Apply the action using your game implementation
        # Calculate the reward and whether the game is done

        action[self.game.blocked_fields] = 0
        index = np.unravel_index(
            np.argmax(action), action.shape)  # game_idx, field_idx

        win_state = self.game.play(*index)
        new_state = self.get_state()
        reward = int(win_state[0]) + int(win_state[1]) * 5
        done = self.game.done

        return new_state, reward, done, {}

    def render(self):
        # Optionally implement this method for rendering the game state
        print(self.game)

    def get_last_move_matrix(self):
        last_move = self.game.last_move
        last_move_matrix = np.zeros((9, 9), dtype=bool)
        if last_move:
            last_move_matrix[last_move] = True
        return last_move_matrix

    def get_state(self):
        white_board = self.game.white.board
        black_board = self.game.black.board
        legal_moves = ~self.game.blocked_fields
        last_move = self.get_last_move_matrix()

        return np.stack([white_board, black_board, legal_moves, last_move])


def random_policy(state):
    white_board, black_board, legal_moves, last_move = tuple(
        state[i] for i in range(4))
    move_probabilities = np.where(
        legal_moves, np.random.random(81).reshape((9, 9)), 0)
    return move_probabilities
'''