from agents.agent import Agent
from environments.game import Game


class Human(Agent):
    """
    Class representing a human player agent.

    Attributes:
        None

    """
    def __init__(self):
        """
        Initialize the Human player agent.

        Args:
            None

        """
        super().__init__()

    def play(self, game: Game) -> tuple:
        """
        Get a move from the human player.

        The function prompts the user to input their move until a valid move is provided.

        Args:
            game (Game): The current game instance.

        Returns:
            tuple: Represents the human player's move.

        """
        valid_move = False
        while not valid_move:
            # (e.g., '2 3', or '23', or '2,3')
            inpt = input("Enter your move! >> ")
            nums = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            move = []
            counter = 0
            for c in inpt:
                if c in nums:
                    counter += 1
                    move.append(int(c)-1)
            if counter < 2:
                continue
            move = (move[0], move[1])
            if game.check_valid_move(*move):
                break
            print(f"{move} is not a valid move")

        return move
