from random import sample
from agents.agent import Agent
from environments.game import Game


class Random(Agent):
    def __init__(self):
        '''
        Initialize a Random agent.

        This agent makes random moves during gameplay.

        '''
        super().__init__()

    def play(self, game: Game) -> tuple:
        '''
        Choose a random valid move for the current game state.

        Args:
            game (Game): The current game state.

        Returns:
            tuple: The chosen move as a tuple.

        '''
        valid_moves = game.get_valid_moves()
        
        try:
            move = sample(sorted(valid_moves), 1)[0]
        except ValueError:
            # Handle the case where there are no valid moves
            print(valid_moves)
            print(game) 
        
        return move