# normal DDPG

import numpy as np
import gym
from ddpg import DDPG
# import matplotlib.pyplot as plt
import pickle

env = gym.make('HalfCheetah-v2')

agent = DDPG(a_dim=6, s_dim=17, a_bound=1)
exploration_rate = 0.2
np.random.seed(1)
env.seed(1)

total_reward = []
for episode in range(1000):
    state = env.reset()
    var = 0.1
    cum_reward = 0
    for step in range(1000):
        if np.random.uniform() > exploration_rate:
            action = np.clip(np.random.normal(np.reshape(agent.choose_action(state), [6, ]), var), -1, 1)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # print(action)
        cum_reward += reward
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        grad = agent.learn()
        if done:
            print('Episode', episode, ' Complete at reward ', cum_reward, '!!!')
            # print('Final velocity x is ',state[9])
            # print('Final velocity z is ',state[10])
            break
        if step == 1000 - 1:
            print('Episode', episode, ' finished at reward ', cum_reward)
            # print('Final velocity x is ', state[9])
            # print('Final velocity z is ', state[10])
    total_reward.append(cum_reward)

# plt.plot(total_reward)
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.title('MountainCar continuous')
# plt.savefig('ddpg')
pickle.dump(total_reward, open('DDPG', 'wb'))
