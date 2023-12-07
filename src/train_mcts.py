from agents.mcts import MCTS

'''
Creates memory file for MCTS.
Updates memory when re-running.
NOTE: needs a LOT of iterations to achieve good perfomance, due to tree size,
and therefore file size gets large quickly...
'''

if __name__ == "__main__":
    mcts = MCTS(memory_path="data/models/mcts/mcts_memory_small.pkl")
    mcts.train(num_iterations=10_000, max_time=100)

