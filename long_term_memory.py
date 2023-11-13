import pickle
from agents import mcts
from environments.game import Game
from agents.mcts import MCTS

train = False

# training
if train == True:
    game = Game()

    mcts = MCTS()

    mcts.train(num_iterations=1000000000, max_time=30)

else:
    print("load mcts memory")
    with open("data/mcts_ltmm.pkl", 'rb') as file:
        memory = pickle.load(file)
    game = Game()
    print("build mcts agent")
    mcts = MCTS(memory=memory)

    print("make first move")
    move = mcts.play(game=game, num_iterations=10, disable_progress_bar=False)

    print(move)

