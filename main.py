
from gym_envs.test_env import UltimateTicTacToeEnv, random_policy

if __name__ == "__main__":
    
    env = UltimateTicTacToeEnv()
    state = env.reset()

    reward_history_white = []
    reward_history_black = []

    done = False
    counter = 0
    while not done:
        action = random_policy(state)
        state, reward, done, _ = env.step(action)

        if counter % 2:
            reward_history_black.append(reward)
        else:
            reward_history_white.append(reward)

        env.render()

        print("")
        print("-"*31)
        print("")
        counter = counter + 1
    print(f"white rewards: {reward_history_white}", sum(reward_history_white))
    print(f"black rewards: {reward_history_black}", sum(reward_history_black))
    