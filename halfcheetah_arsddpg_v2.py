from integrated_ARS import *
import gym
# import matplotlib.pyplot as plt
# explore the difference between actions
import pickle

coefficients = [0.01, 0.02, 0.05, 0.1, 0.5]
for i in range(len(coefficients)):
    hp = HP(env=gym.make('HalfCheetah-v2'), noise=0.03, nb_steps=1000, episode_length=1000, num_deltas=16,
            num_best_deltas=16, learning_rate=0.02, seed=1, namespace='arsddpg_' + str(i),ddpg_normalized_parameter=coefficients[i])
    trainer = ARSTrainer(hp=hp, monitor_dir=None, input_size=17, output_size=6)
    reward = trainer.train()
    pickle.dump(reward, open('halfcheetah_arsddpg_' + str(i)+'_coefficients_'+str(coefficients[i]), 'wb'))
