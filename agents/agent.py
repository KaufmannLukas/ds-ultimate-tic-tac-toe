from abc import ABC, abstractmethod
from environments.game import Game

class Agent(ABC):

    def __init__(self):
        super().__init__()

    def play(game: Game) -> tuple:
        # make a move according to specific game state / board position
        pass


# prepare / set up agent for different games / environments?
    


