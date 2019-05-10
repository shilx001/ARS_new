from integrated_ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(len(seeds)):
    hp = HP(env=gym.make('Ant-v2'), noise=0.025, nb_steps=1000, episode_length=1000, num_deltas=60,
            num_best_deltas=20, learning_rate=0.015, seed=seeds[i], namespace='arsddpg_' + str(i), ddpg_step=100,
            ddpg_normalized_parameter=1.0/100)
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=111, output_size=8)
    reward = trainer.train()
    pickle.dump(reward, open('ant_arsddpg_v0_seeds_' + str(seeds[i]), 'wb'))
