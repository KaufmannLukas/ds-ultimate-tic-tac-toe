import pickle
from agents import mcts
from environments.game import Game
from agents.mcts import MCTS

train = True

# training
if train == True:
    game = Game()

    mcts = MCTS(memory_path="data/mcts_ltmm_02.pkl", update_memory=True)

    mcts.train(num_iterations=10_000_000, max_time=None)

# else:
#     print("load mcts memory")
#     with open("data/mcts_ltmm_02.pkl", 'rb') as file:
#         memory = pickle.load(file)
#     game = Game()
#     print("build mcts agent")
#     mcts = MCTS(memory=memory)

#     print("make first move")
#     move = mcts.play(game=game, num_iterations=10, disable_progress_bar=False)

#     print(move)

