from random import sample
from agents.agent import Agent
from environments.game import Game


class Random(Agent):
    def __init__(self):
        super().__init__()

    def play(self, game: Game) -> tuple:
        valid_moves = game.get_valid_moves()
        move = sample(sorted(valid_moves), 1)[0]
        return move