import os
import numpy as np
import gym
from gym import wrappers


class HP:
    # Hyperparameters
    def __init__(self,
                 nb_steps=1000,
                 episode_length=1000,
                 learning_rate=0.02,
                 num_deltas=16,
                 num_best_deltas=16,
                 noise=0.03,
                 seed=1,
                 env=None,
                 record_every=50):
        self.nb_steps = nb_steps  # rollout number
        self.episode_length = episode_length  # 每个episode的长度
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas  # delta的个数
        self.num_best_deltas = num_best_deltas  # 最优delta的个数
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise  # 噪音
        self.seed = seed  # random seed
        self.env = env
        self.record_every = record_every


class Normalizer:
    # Normalizes the input observations
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):  # observe a space, dynamic implementation of calculate mean of variance
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)  # 方差，清除了小于1e-2的

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class Policy:
    def __init__(self, input_size, output_size, hp):
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:  # 如果没标明方向则返回theta*input(矩阵乘)
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(input)  # 如果是+方向则返回
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]  # 针对每个采样正态分布采样

    def update(self, rollouts, sigma_rewards):  # 针对rollout中的值对policy进行update
        # sigma_rewards is the standard deviation of the rewards
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step


class ARSTrainer:
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 policy=None,
                 monitor_dir=None):

        self.hp = hp or HP()
        np.random.seed(self.hp.seed)
        self.env = self.hp.env
        self.env.seed(self.hp.seed)
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.hp.episode_length = self.hp.episode_length
        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]
        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = policy or Policy(self.input_size, self.output_size, self.hp)
        self.record_video = False

    # Explore the policy on one specific direction and over one episode
    def explore(self, direction=None, delta=None, print_done=False):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < self.hp.episode_length:  # exectution
            self.normalizer.observe(state)  # 观测到一个state
            state = self.normalizer.normalize(state)  # 对state进行Normalize
            action = self.policy.evaluate(state, delta, direction)  # 执行
            state, reward, done, _ = self.env.step(action)
            #reward = max(min(reward, 1), -1)#reward被规则化到0，1之间，最大为1,最小为-1
            sum_rewards += reward
            num_plays += 1
        if done and print_done:
            #print('Reach the goal!')
            pass
        return sum_rewards

    def train(self):
        return_reward = []
        for step in range(self.hp.nb_steps):
            # initialize the random noise deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas()
            positive_rewards = [0] * self.hp.num_deltas
            negative_rewards = [0] * self.hp.num_deltas

            # play an episode each with positive deltas and negative deltas, collect rewards
            for k in range(self.hp.num_deltas):
                positive_rewards[k] = self.explore(direction="+", delta=deltas[k])
                negative_rewards[k] = self.explore(direction="-", delta=deltas[k])

            # Compute the standard deviation of all rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.hp.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update the policy
            self.policy.update(rollouts, sigma_rewards)

            # Only record video during evaluation, every n steps
            if step % self.hp.record_every == 0:
                self.record_video = True
            # Play an episode with the new weights and print the score
            reward_evaluation = self.explore(print_done=True)
            print('Step: ', step, 'Reward: ', reward_evaluation)
            return_reward.append(reward_evaluation)
            self.record_video = False
        return return_reward.copy()
