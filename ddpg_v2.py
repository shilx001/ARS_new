import gym
import numpy as np
import tensorflow as tf


# linear DDPG with ARS update
class DDPG:
    def __init__(self, a_dim, s_dim, a_bound,
                 lr_a=0.0001,
                 lr_c=0.0001,
                 gamma=0.99,
                 memory_capacity=1000000,
                 batch_size=128,
                 hidden_size=64,
                 replay_start=100,
                 std_dev=0.01,
                 tau=0.001,
                 seed=1,
                 namespace="default"):
        self.lr_a, self.lr_c, self.gamma = lr_a, lr_c, gamma
        self.memory_capacity, self.batch_size, self.hidden_size, self.replay_start, self.std_dev = \
            memory_capacity, batch_size, hidden_size, replay_start, std_dev
        self.tau = tau
        self.seed = seed
        self.namespace=namespace
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 2), dtype=np.float64)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float64, [None, s_dim], name='S')
        self.S_ = tf.placeholder(tf.float64, [None, s_dim], name='S_')
        self.R = tf.placeholder(tf.float64, [None, ], name='reward')
        self.a = self._build_a(self.S)
        self.done = tf.placeholder(tf.float64, [None, ], name='done')
        q = self._build_c(self.S, self.a)
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor_'+self.namespace)
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic_'+self.namespace)

        a_ = self._build_a(self.S_, reuse=True)
        q_ = self._build_c(self.S_, a_, reuse=True)

        a_loss = - tf.reduce_mean(q)  # maximize the q, 这里我改为reduce_sum
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.a_params)
        self.a_gradients = tf.gradients(a_loss, self.a_params)
        q_target = self.R + (1 - self.done) * self.gamma * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def learn(self):
        # return the calculated gradient of the actor network
        if self.pointer < self.replay_start:
            return
        if self.pointer < self.memory_capacity:
            indices = np.random.choice(self.pointer, size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]  # transitions data
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim - 1:-1]
        bdone = bt[:, -1]
        self.sess.run(self.atrain, feed_dict={self.S: bs})
        self.sess.run(self.ctrain, feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.done: bdone})
        grad = self.sess.run(self.a_gradients, feed_dict={self.S: bs})
        return np.reshape(grad, [self.s_dim, self.a_dim])

    def store_transition(self, s, a, r, s_, done):  # 每次存储一个即可
        s = np.reshape(np.array(s), [self.s_dim, 1])
        a = np.reshape(np.array(a), [self.a_dim, 1])
        r = np.reshape(np.array(r), [1, 1])
        s_ = np.reshape(np.array(s_), [self.s_dim, 1])
        done = np.reshape(np.array(done), [1, 1])

        transition = np.vstack((s, a, r, s_, done))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = np.reshape(transition, [2 * self.s_dim + self.a_dim + 2, ])
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor_'+self.namespace, reuse=reuse, custom_getter=custom_getter):
            w1 = tf.Variable(np.zeros([self.s_dim, self.a_dim]), trainable=trainable)
            return tf.matmul(s, w1)

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic_'+self.namespace, reuse=reuse, custom_getter=custom_getter):
            input_s = tf.reshape(s, [-1, self.s_dim])
            input_a = tf.reshape(a, [-1, self.a_dim])
            input_all = tf.concat([input_s, input_a], axis=1)  # s: [batch_size, s_dim]
            h1 = tf.layers.dense(input_all, units=self.hidden_size, activation=tf.nn.relu, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=self.std_dev))
            h2 = tf.layers.dense(h1, units=self.hidden_size, activation=tf.nn.relu, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=self.std_dev))
            h3 = tf.layers.dense(h2, units=1, activation=None, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=self.std_dev))
            return h3

    def memory_flush(self):
        self.memory = np.zeros((self.memory_capacity, self.s_dim * 2 + self.a_dim + 2), dtype=np.float64)

    def update_actor(self, value):
        # update the paramters of actor using ARS
        update_op = tf.assign(self.a_params[0], value)
        self.sess.run(update_op)
        self.w1_value = self.sess.run(self.a_params)
        # print(self.w1_value)

    def choose_action(self, s):
        s = np.reshape(s, [-1, self.s_dim])
        return self.sess.run(self.a, feed_dict={self.S: s})
