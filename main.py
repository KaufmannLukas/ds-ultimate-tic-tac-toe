
from gym_envs.test_env import UltimateTicTacToeEnv, random_policy

if __name__ == "__main__":
    
    env = UltimateTicTacToeEnv()
    state = env.reset()

    done = False
    while not done:
        action = random_policy(state)
        state, reward, done, _ = env.step(action)
        env.render()

        print("")
        print("-"*31)
        print("")