from ARS import *
import gym
# import matplotlib.pyplot as plt
import pickle

seeds = [1, 2, 3, 4, 5]

for i in range(len(seeds)):
    hp = HP(env=gym.make('HalfCheetah-v2'), noise=0.03, nb_steps=1000, episode_length=1000, num_deltas=16,
            num_best_deltas=16, learning_rate=0.02, seed=seeds[i])
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=17, output_size=6)
    reward = trainer.train()
    pickle.dump(reward, open('halfcheetah_ars_'+str(i), 'wb'))
