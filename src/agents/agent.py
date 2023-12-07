from abc import ABC, abstractmethod
from environments.game import Game


class Agent(ABC):
    """
    Abstract base class for defining game-playing agents.

    Attributes:
        None

    """
    def __init__(self):
        """
        Initialize the Agent.

        Args:
            None

        """    
        super().__init__()


    @abstractmethod
    def play(self, game: Game) -> tuple:
        """
        Abstract method for making a move according to specific game state / board position.
        @abstractmethod decorator indicates that any concrete subclass must provide an implementation for the play method

        Args:
            game (Game): The current game instance.

        Returns:
            tuple: Represents the agent's move.

        """
        pass

