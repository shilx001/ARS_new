from integrated_ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(len(seeds)):
    hp = HP(env=gym.make('Hopper-v2'), noise=0.02, nb_steps=1000, episode_length=1000, num_deltas=4,
            num_best_deltas=4, learning_rate=0.02, seed=seeds[i], namespace='arsddpg_' + str(i))
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=11, output_size=3)
    reward = trainer.train()
    pickle.dump(reward, open('hopper_arsddpg_seeds_' + str(seeds[i]), 'wb'))
