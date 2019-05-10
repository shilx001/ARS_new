from ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(len(seeds)):
    hp = HP(env=gym.make('Walker2d-v2'), noise=0.025, nb_steps=1000, episode_length=1000, num_deltas=40,
            num_best_deltas=30, learning_rate=0.03, seed=seeds[i])
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=17, output_size=6)
    reward = trainer.train()
    pickle.dump(reward, open('walker2d_ars_v0_seeds_' + str(seeds[i]), 'wb'))

