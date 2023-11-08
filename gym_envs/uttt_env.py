import gymnasium as gym
from gymnasium import spaces
from matplotlib.pylab import get_state
import numpy as np

from environments.game import Game


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
