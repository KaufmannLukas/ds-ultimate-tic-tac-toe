from agents.mcts import MCTS


if __name__ == "__main__":
    mcts = MCTS(memory_path="data/models/mcts/test.pkl")
    mcts.train(num_iterations=1000, max_time=100)

