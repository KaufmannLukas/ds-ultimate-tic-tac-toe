from random import choice
import logging
from anyio import open_process
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib.pylab import get_state

from environments.game import Game
from agents.agent import Agent

from typing import Optional


logger = logging.getLogger(__name__)


class UltimateTicTacToeEnv(gym.Env):
    def __init__(self, opponent: Optional[Agent] = None, opponent_starts=False, reward_config=None):
        logger.info("init env")
        super(UltimateTicTacToeEnv, self).__init__()

        self.reward_config = reward_config

        self.action_space = spaces.Box(
            low=0, high=1, shape=(9, 9), dtype=np.float64)  # 9x9 board

        # 9x9 game boards for white, black, last_move, blocked_field
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(4, 9, 9), dtype=bool)
        
        self.game = Game()

        self.full_reward_history = []
        self.single_reward_history = []


        self.opponent = opponent
        self.opponent_starts = opponent_starts
    
        self.reset()

    def reset(self):
        self.full_reward_history.append(self.single_reward_history)
        self.single_reward_history = []
        #logger.info("reset env")
        self.game = Game()
        if self.opponent is not None and self.opponent_starts:
                opponent_move = self.opponent.play(self.game)
                self.game.play(*opponent_move)
        return game2tensor(self.game), {}
        

    def step(self, action: int):
        '''
        action: tuple like (game_idx, field_idx)
        '''
        
        '''
        Reward factors:
        '''
        if self.reward_config is not None:


            global_win_factor = self.reward_config["global_win_factor"]
            global_draw_factor = self.reward_config["global_draw_factor"]

            local_win_factor = self.reward_config["local_win_factor"]
            local_draw_factor = self.reward_config["local_draw_factor"]

            legal_move_factor = self.reward_config["legal_move_factor"]
            illegal_move_factor = self.reward_config["illegal_move_factor"]
        else:
            global_win_factor = 100
            global_draw_factor = 10

            local_win_factor = 5
            local_draw_factor = 2

            legal_move_factor = 1
            illegal_move_factor = -2


        
        reward = 0
        global_draw, local_draws = self.game.check_draw()

        game_idx = action // 9
        field_idx = action % 9

        move = (game_idx, field_idx)

        if self.game.check_valid_move(*move):
            reward += 1 * legal_move_factor
            local_win, global_win = self.game.play(*move)
            reward += int(local_win) * local_win_factor + int(global_win) * global_win_factor

            if self.opponent is not None and not self.game.done:
                counter_action = self.opponent.play(self.game)
                for i in range(100):
                    if self.game.check_valid_move(*counter_action):
                        self.game.play(*counter_action)
                        break
                    else:
                        counter_action = self.opponent.play(self.game)
                else:
                    self.game = Game()
                    logger.warn("opponent is not making valid moves. Restart the Game!!!")

        else:
            reward += 1 * illegal_move_factor
            self.game = Game()

        new_global_draw, new_local_draws = self.game.check_draw()
        reward += int(new_global_draw) * global_draw_factor
        reward += int(any(new_local_draws ^ local_draws)) * local_draw_factor

        new_state = game2tensor(self.game)
        done = self.game.done

        # Append reward history => needed to implement numbers into csv file in 
        self.single_reward_history.append(reward)

        return new_state, reward, done, {}, {}


    def render(self):
        print(self.game)

def game2tensor(game: Game):

    wb = game.white.board
    bb = game.black.board
    bf = game.blocked_fields
    vm = ~game.blocked_fields
    lm = np.zeros(shape=(9, 9), dtype=bool)
    if game.last_move:
        lm[*game.last_move] = True

    if game.current_player.color == "white (X)":
        res = np.stack([wb, bb, vm, lm])
    else:
        res = np.stack([bb, wb, vm, lm])

    return res.flatten()
