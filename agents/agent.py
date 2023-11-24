from abc import ABC, abstractmethod
from environments.game import Game


class Agent(ABC):

    def __init__(self):
        super().__init__()

    def play(self, game: Game) -> tuple:
        # make a move according to specific game state / board position
        pass
