from itertools import permutations



from agents.human import Human
from agents.mcts import MCTS
from agents.ppo import PPO
from agents.random import Random
from agents.agent import Agent
from environments.game import Game


def play(white: Agent, black: Agent):
    game = Game()

    counter = 0
    while not game.done:
        if counter % 2 == 0:
            next_move = white.play(game)
        else:
            next_move = black.play(game)
        game.play(*next_move)
        counter += 1

    return game

def tournament(agents, rounds = 10):
    perms = permutations(agents, r=2)

    for agent_1, agent_2 in perms:
        print(f"{agent_1} as white vs. {agent_2} as black")
        for i in range(rounds):
            res = play(agent_1, agent_2)
            print(f"round {i}: {res.winner}")


if __name__ == "__main__":
    # create agents
    agent_1 = Human()
    agent_2 = Random()
    agent_3 = MCTS(num_iterations=100, memory_path="data/models/mcts/test.pkl")

    model = "ppo_test"
    path = "./data/models/ppo/"
    agent_4 = PPO(name=model, path=path, helper=agent_2)


    play(agent_1, agent_3)

    #tournament([agent_2, agent_3, agent_4], rounds=5)
