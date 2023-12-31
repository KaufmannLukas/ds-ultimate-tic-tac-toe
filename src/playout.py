from collections import defaultdict
from itertools import permutations
import logging
import time
from datetime import datetime
import csv

from agents.human import Human
from agents.mcts import MCTS
from agents.ppo import PPO
from agents.random import Random
from agents.agent import Agent
from environments.game import Game


# Format the date and time as a string with seconds precision and no spaces
formatted_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=f"logs/playout_{formatted_date_time}.log",
                    filemode='w'
                    )
logger = logging.getLogger(__name__)


def play(white: Agent, black: Agent) -> Game:
    """
    Simulate a game between two agents.

    Parameters:
    - white (Agent): The first player agent.
    - black (Agent): The second player agent.

    Returns:
    - Game: The result of the game.
    """
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


def tournament(agents, names, rounds=10):
    """
    Conduct a tournament between agents.

    Parameters:
    - agents (list): List of agent instances participating in the tournament.
    - names (list): List of names corresponding to the agents.
    - rounds (int): Number of rounds for each matchup.

    Writes results to logs and a CSV file.
    """
    perms = permutations(zip(agents, names), r=2)
    results = []
    summary = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})

    for agent_1, agent_2 in perms:
        logger.info("-" * 20)
        logger.info(f"{agent_1[1]} as white vs. {agent_2[1]} as black")
        for i in range(rounds):
            res = play(agent_1[0], agent_2[0])
            logger.info(f"round {i}: {res.winner}")

            # Update results
            results.append([agent_1[1], agent_2[1], i, res.winner])

            # Update summary
            if res.winner == "white (X)":
                summary[(agent_1[1], agent_2[1])]['wins'] += 1
            elif res.winner == "black (O)":
                summary[(agent_1[1], agent_2[1])]['losses'] += 1
            else:  # Draw
                summary[(agent_1[1], agent_2[1])]['draws'] += 1

    # Logging the summary
    logger.info("Tournament Summary:")
    for agents, result in summary.items():
        logger.info(f"{agents[0]} vs {agents[1]}: Wins: {result['wins']}, Losses: {result['losses']}, Draws: {result['draws']}")

    # Writing results to CSV
    with open(f'data/local/playout_results_{formatted_date_time}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['White', 'Black', 'Round', 'Winner'])
        writer.writerows(results)


if __name__ == "__main__":
    # Create agents
    agent_1 = Human()
    agent_2 = Random()
    agent_3 = MCTS(num_iterations=5, memory_path="data/models/mcts/mcts_memory_small.pkl")

    model = "ppo_v1"
    path = "./data/models/ppo/"
    agent_4 = PPO(name=model, path=path, helper=agent_2)

    # Tournament with specified agents
    tournament([agent_2, agent_3, agent_4], 
               names=["random", "mcts", "ppo"], 
               rounds=10)
    

    # NOTE: if you want to play, here is an example:
    # play(agent_1, agent_3)
