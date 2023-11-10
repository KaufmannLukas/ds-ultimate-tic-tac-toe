from agents.agent import Agent
from environments.game import Game


class Human(Agent):

    def __init__(self):
        super().__init__()

    def play(self, game: Game) -> tuple:
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
