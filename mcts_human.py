
from environments.game import Game
from agents.mcts import MCTS
from agents.human import Human

if __name__ == "__main__":
    game = Game()
    computer_agent = MCTS()
    human_agent = Human()

    counter = 0
    while not game.done:
        print("-"*31)
        print(game)
        print("-"*31)
        if counter % 2 == 0:
            next_move = human_agent.play(game)
        else:
            next_move = computer_agent.play(game, num_iterations=1000)
        print(f"last move: ({next_move[0]+1}, {next_move[1]+1})")
        game.play(*next_move)
        counter += 1

    print(game)
