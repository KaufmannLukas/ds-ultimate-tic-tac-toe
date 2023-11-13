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
    with open("data/mcts_ltmm.pkl", 'rb') as file:
        memory = pickle.load(file)
    game = Game()
    mcts = MCTS(memory=memory)

    move = mcts.play(game=game)

    print(move)

