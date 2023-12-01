from random import sample
from agents.agent import Agent
from environments.game import Game


class Random(Agent):
    def __init__(self):
        super().__init__()

    def play(self, game: Game) -> tuple:
        valid_moves = game.get_valid_moves()
        try:
            move = sample(sorted(valid_moves), 1)[0]
        except ValueError:
            print(valid_moves)
            print(game) 
        return move
