import gym
import random
import numpy as np

env = gym.make('SpaceInvaders-ram-v0')
action_space_size = env.action_space.n

num_episodes = 10000
max_steps_per_ep = 100

l_rate = .1
d_rate = .99

expl_rate = 1
max_expl_rate = 1
min_expl_rate = 0.01
expl_dec_rate = 0.001

rewards_all_episodes = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_curr_ep = 0
    print("hey")
    while(not done):

        action = env.action_space.sample()

        print(state)
        new_state, reward, done, info = env.step(action)

        state = new_state
        rewards_curr_ep += reward

    rewards_all_episodes.append(rewards_curr_ep)

rewards_per_thous = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("**AVG REWARD**")
for r in rewards_per_thous:
    print(count, ": ", str(sum(r/1000)))
    count += 1000


