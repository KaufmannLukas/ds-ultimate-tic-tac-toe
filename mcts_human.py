import pickle

from environments.game import Game
from agents.mcts import MCTS
from agents.human import Human
import pickle


if __name__ == "__main__":
    game = Game()
    #implement the next two lines for using a memory_prone agent (like mcts_agent_01)
    # with open("data/mcts_ltmm_02.pkl", mode="rb") as file:
    #     memory = pickle.load(file)
    #change the agent to play against here:
    # computer_agent = MCTS(memory_path = "data/mcts_ltmm_02.pkl")
    computer_agent = MCTS()
    human_agent = MCTS()

    # TODO: improve interface (print local/global wins, etc.)
    # TODO: implement game loop (replay)
    counter = 0
    while not game.done:
        print("-"*31)
        print(game)
        print("-"*31)
        if counter % 2 != 0:
            next_move = human_agent.play(game, num_iterations=1000)
        else:
            next_move = computer_agent.play(game, num_iterations=1000)
        print(f"last move: ({next_move[0]+1}, {next_move[1]+1})")
        result = game.play(*next_move)
        print(result)
        counter += 1

        
    json_string = game.make_json()
    with open("json_test_done_new.json", "w") as file:
        file.write(json_string)


    print(game)
