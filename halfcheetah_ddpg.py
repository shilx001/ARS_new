# normal DDPG

import numpy as np
import gym
from ddpg import DDPG
# import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

env = gym.make('HalfCheetah-v2')

seeds=[1,2,3,4,5]

for i in range(len(seeds)):
    agent = DDPG(a_dim=6, s_dim=17, a_bound=1, lr_a=0.0001,lr_c=0.001, seed=seeds[i],namespace='ddpg_'+str(i))
    exploration_rate = 0.2
    np.random.seed(seeds[i])
    env.seed(seeds[i])
    tf.set_random_seed(seeds[i])

    total_reward = []
    for episode in range(1000):
        state = env.reset()
        var = 0.1
        cum_reward = 0
        for step in range(1000):
            # action = np.clip(np.random.normal(np.reshape(agent.choose_action(state), [6, ]), var), -1, 1)
            if np.random.uniform() > exploration_rate:
                action = np.clip(np.random.normal(np.reshape(agent.choose_action(state), [6, ]), var), -1, 1)
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            # print(action)
            cum_reward += reward
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()
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
    #pickle.dump(total_reward, open('halfcheetah_ddpg_'+str(i), 'wb'))
