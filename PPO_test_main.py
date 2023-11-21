from agents.ppo import PPO
from agents.random_agnt import  Random
from agents.human import Agent
from gym_envs.uttt_env import UltimateTicTacToeEnv
from environments.game import Game
from agents.human import Human

#from agents.network import FeedForwardNN

if __name__ == "__main__":

    # Agent = Agent / Random ?

    random_opponent = Random()

    env = UltimateTicTacToeEnv(opponent=random_opponent, opponent_starts=False)


    print(env.opponent)
    # Create a model for PPO.
    model = PPO(env=env)
    # model = PPO(policy_class=FeedForwardNN, env=env)
    #model = PPO(env)
    model.learn(20000)

    game = Game()
    #implement the next two lines for using a memory_prone agent (like mcts_agent_01)
    # with open("data/mcts_ltmm_02.pkl", mode="rb") as file:
    #     memory = pickle.load(file)
    #change the agent to play against here:
    computer_agent = model
    human_agent = Human()

    # TODO: improve interface (print local/global wins, etc.)
    # TODO: implement game loop (replay)
    counter = 0
    while not game.done:
        print("-"*31)
        print(game)
        print("-"*31)
        if counter % 2 != 0:
            next_move = human_agent.play(game)
        else:
            next_move = computer_agent.play(game)
        print(f"last move: ({next_move[0]+1}, {next_move[1]+1})")
        result = game.play(*next_move)
        print(result)
        counter += 1

    print(game)

