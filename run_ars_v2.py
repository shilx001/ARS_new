from integrated_ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle
import os

# hp = HP(env=MountainCar(), noise=0.3, nb_steps=500)
hp = HP(env=gym.make('Ant-v2'), noise=0.03, nb_steps=1000, episode_length=1000, num_deltas=16,
        num_best_deltas=16, learning_rate=0.02)
trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=111, output_size=8)
reward = trainer.train()
pickle.dump(reward, open('ant_arsddpg_v0', 'wb'))
# plt.plot(reward)
# plt.savefig('Shaped_reward_FetchReach')
