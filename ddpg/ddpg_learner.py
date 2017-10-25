import numpy as np
import tensorflow as tf
import os
import pickle

target_update_step = 0.001
MIN_REPLAY_SIZE = 5000
REG_STEP = 0.00001


def dense(shape, name):
    # cur_w = tf.Variable(tf.truncated_normal(shape, -0.1, 0.1), name=name + "_w")
    # cur_b = tf.Variable(tf.truncated_normal([shape[1]], -0.1, 0.1), name=name + "_b")
    # cur_b = tf.Variable(tf.random_normal([shape[1]], stddev=0.01), name=name + "_b")
    cur_w = tf.Variable(tf.random_normal(shape, stddev=0.01), name=name + "_w")
    cur_b = tf.Variable(tf.zeros([shape[1]], dtype=tf.float32), name=name + "_b")
    return cur_w, cur_b


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    def __init__(self, max_size, state_size, action_size):
        self.max_size = max_size
        self.items = []
        self.state_size = state_size
        self.action_size = action_size
        self.full_size = state_size * 2 + action_size + 2
        self.frozen_items = None
        self.frozen_arr_idx = None

    def add(self, item):
        cur = np.zeros((self.full_size,))
        cur[0:self.state_size] = item[0]
        cur[self.state_size:self.state_size + self.action_size] = item[1]
        cur[self.state_size + self.action_size] = item[2]
        cur[self.state_size + self.action_size + 1:-1] = item[3]
        cur[-1] = item[4]
        self.items.append(cur)
        if len(self.items) >= MIN_REPLAY_SIZE:
            if self.frozen_items is None:
                self.frozen_items = np.array(self.items)
            else:
                self.frozen_items = np.concatenate((self.frozen_items, np.array(self.items)), axis=0)
            self.frozen_arr_idx = np.arange(0, self.frozen_items.shape[0]).astype(np.int32)
            self.items = []
        if self.frozen_items is not None and len(self.frozen_items) > self.max_size:
            self.frozen_items = self.frozen_items[50000:]
            self.frozen_arr_idx = np.arange(0, self.frozen_items.shape[0]).astype(np.int32)

    def get_batch(self, batch_size):
        idx = np.random.choice(self.frozen_arr_idx, batch_size, False)
        arr = self.frozen_items[idx, :]
        s_batch = arr[:, 0:self.state_size]
        a_batch = arr[:, self.state_size:(self.state_size + self.action_size)]
        r_batch = arr[:, self.state_size + self.action_size]
        s1_batch = arr[:, (self.state_size + self.action_size + 1):-1]
        t_batch = arr[:, -1]
        return s_batch, a_batch, r_batch, s1_batch, t_batch


class Actor:
    def __init__(self, x_input, state_size, action_size, learning_rate):
        self.x_input = x_input
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.variable_scope("actor_actual") as scope:
            self.actor_net = self.create_actor()
        with tf.variable_scope("actor_target") as scope:
            self.target_net = self.create_actor()
        self.actor_trainable = [x for x in tf.trainable_variables() if x.name.startswith("actor_actual")]
        self.target_trainable = [x for x in tf.trainable_variables() if x.name.startswith("actor_target")]
        self.gradients_placeholder = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.init_optimizer()
        self.init_update_target()

    def create_actor(self):
        d1_w, d1_b = dense([self.state_size, 512], "d1")
        l1 = tf.nn.relu(tf.matmul(self.x_input, d1_w) + d1_b)
        l1 = tf.layers.batch_normalization(l1)
        d2_w, d2_b = dense([512, 512], "d2")
        l2 = tf.nn.relu(tf.matmul(l1, d2_w) + d2_b)
        l2 = tf.layers.batch_normalization(l2)
        d3_w, d3_b = dense([512, 512], "d3")
        l3 = tf.nn.relu(tf.matmul(l2, d3_w) + d3_b)
        l3 = tf.layers.batch_normalization(l3)
        d4_w, d4_b = dense([512, self.action_size], "d4")
        actor_out = tf.clip_by_norm(tf.nn.tanh(tf.matmul(l3, d4_w) + d4_b), 1)
        return actor_out

    def init_update_target(self):
        self.update_target = [self.target_trainable[i].assign
                              (self.target_trainable[i] * (1.0 - target_update_step) + self.actor_trainable[
                                  i] * target_update_step)
                              for i in range(len(self.actor_trainable))]

    def init_optimizer(self):
        grads = tf.gradients(self.actor_net, self.actor_trainable, -self.gradients_placeholder)
        clipped_grads = []
        for i, g in enumerate(grads):
            clipped = tf.clip_by_value(g, -10.0, 10.0)
            regularized = clipped + tf.reduce_sum(self.actor_trainable[i]) * REG_STEP
            clipped_grads.append(regularized)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(clipped_grads, self.actor_trainable))


class Critic:
    def __init__(self, labels, x_input, state_size, action_size, learning_rate):
        self.x_input = x_input
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.labels = labels
        self.actions_in = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        with tf.variable_scope("critic_actual") as scope:
            self.critic_net = self.create_critic(self.actions_in)
        with tf.variable_scope("critic_target") as scope:
            self.target_net = self.create_critic(self.actions_in)
        self.critic_trainable = [x for x in tf.trainable_variables() if x.name.startswith("critic_actual")]
        self.target_trainable = [x for x in tf.trainable_variables() if x.name.startswith("critic_target")]
        self.update_target = None
        self.loss, self.optimizer = self.init_optimizer()
        self.action_gradients = tf.gradients(self.critic_net, self.actions_in)
        self.init_update_target()

    def create_critic(self, actor_out):
        # actor_d0_w, actor_d0_b = dense([self.action_size, 256], "actor_d0")
        # actor_l0 = tf.matmul(actor_out, actor_d0_w) + actor_d0_b
        # critic_d0_w, critic_d0_b = dense([self.state_size, 256], "critic_d0")
        # critic_l0 = tf.matmul(self.x_input, critic_d0_w) + critic_d0_b
        # all_input = tf.nn.relu(actor_l0 + critic_l0)

        all_input = tf.concat([self.x_input, actor_out], axis=1)
        d1_w, d1_b = dense([self.state_size + self.action_size, 512], "d1")
        l1 = tf.nn.relu(tf.matmul(all_input, d1_w) + d1_b)
        d2_w, d2_b = dense([512, 512], "d2")
        l2 = tf.nn.relu(tf.matmul(l1, d2_w) + d2_b)
        d3_w, d3_b = dense([512, 512], "d3")
        l3 = tf.nn.relu(tf.matmul(l2, d3_w) + d3_b)
        d4_w, d4_b = dense([512, 1], "d4")
        critic_out = tf.identity(tf.matmul(l3, d4_w) + d4_b)

        #d1_w, d1_b = dense([self.state_size, 512], "d1")
        #d2_w, d2_b = dense([512, 256], "d2")
        #d2_w_action, d2_b = dense([self.action_size, 256], "d2_action")
        #d3_w, d3_b = dense([256, 1], "d3")
        #l1 = tf.nn.relu(tf.matmul(self.x_input, d1_w) + d1_b)
        #l1 = tf.layers.batch_normalization(l1)
        #l2 = tf.nn.relu(tf.matmul(l1, d2_w) + tf.matmul(actor_out, d2_w_action) + d2_b)
        #l2 = tf.layers.batch_normalization(l2)
        #critic_out = tf.identity(tf.matmul(l2, d3_w) + d3_b)
        return critic_out

    def init_update_target(self):
        self.update_target = [self.target_trainable[i].assign
                              (self.target_trainable[i] * (1.0 - target_update_step) + self.critic_trainable[
                                  i] * target_update_step)
                              for i in range(len(self.critic_trainable))]

    def init_optimizer(self):
        loss = tf.reduce_mean(tf.squared_difference(self.critic_net, self.labels))
        for x in self.critic_trainable:
            loss += tf.nn.l2_loss(x) * REG_STEP
        return loss, tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=self.critic_trainable)


class Runner:
    def __init__(self, session, actor, critic, buffer, batch_size, discount, save_path):
        self.session = session
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.batch_size = batch_size
        self.discount = discount
        self.last_loss = 0
        self.save_path = save_path
        self.saver = tf.train.Saver()
        if os.path.exists(save_path + ".index"):
            print("Restoring model")
            self.saver.restore(session, save_path)
            self.buffer = pickle.load(open(self.save_path + "_replay_buffer.pickle", "rb"))
        self.step = 0

    def optimize(self, s_batch, a_batch, y_batch):
        _, cl = self.session.run([self.critic.optimizer, self.critic.loss], feed_dict={
            self.critic.x_input: s_batch,
            self.critic.labels: y_batch,
            self.critic.actions_in: a_batch
        })
        actor_preds = self.session.run([self.actor.actor_net], feed_dict={
            self.actor.x_input: s_batch
        })
        grads = self.session.run([self.critic.action_gradients], feed_dict={
            self.critic.actions_in: actor_preds[0],
            self.critic.x_input: s_batch
        })
        self.session.run([self.actor.optimizer], feed_dict={
            self.actor.gradients_placeholder: grads[0][0],
            self.actor.x_input: s_batch
        })
        return cl

    def predict_actor_target(self, s_batch):
        return self.session.run([self.actor.target_net], feed_dict={
            self.actor.x_input: s_batch
        })[0]

    def predict_actor(self, s_batch):
        return self.session.run([self.actor.actor_net], feed_dict={
            self.actor.x_input: s_batch
        })[0][0]

    def predict_critic_target(self, s_batch, a_batch):
        return self.session.run([self.critic.target_net], feed_dict={
            self.critic.x_input: s_batch,
            self.critic.actions_in: a_batch
        })[0]

    def update_targets(self):
        self.session.run([self.actor.update_target])
        self.session.run([self.critic.update_target])

    def add(self, state, action, reward, state1, terminal):
        batch_item = [state, action, reward, state1, terminal]
        self.buffer.add(batch_item)
        self.last_loss = 0
        if self.step % 100000 == 0:
            print("Saving model...")
            self.saver.save(self.session, self.save_path)
            pickle.dump(self.buffer, open(self.save_path + "_replay_buffer.pickle", "wb"))
            print("Model saved!")
        self.update_targets()
        self.step += 1

    def make_train(self):
        if self.step > MIN_REPLAY_SIZE:
            self.last_loss = self.train_on_batch(self.buffer.get_batch(self.batch_size))

    def get_action(self, state):
        action = self.predict_actor(np.array([state]))
        return action

    def train_on_batch(self, batch):
        s_batch, a_batch, r_batch, s1_batch, t_batch = batch
        a1_batch = self.predict_actor_target(s1_batch)
        target_q = self.predict_critic_target(s1_batch, a1_batch)
        r_b = r_batch.reshape((len(r_batch), 1))
        y = r_b + self.discount * target_q
        t_batch = t_batch.astype(np.bool)
        y[t_batch, 0] = r_batch[t_batch]
        return self.optimize(s_batch, a_batch, y)


def create_runner(save_path, state_dim, action_dim, discount=0.999, buffer_size=100000, batch_size=128):
    g = tf.Graph()
    with g.as_default():
        actor_x_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
        critic_x_input = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
        y_labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        actor = Actor(actor_x_input, state_dim, action_dim, 1e-4)
        critic = Critic(y_labels, critic_x_input, state_dim, action_dim,  1e-3)
    sess = tf.InteractiveSession(graph=g)
    tf.global_variables_initializer().run()
    buf = ReplayBuffer(buffer_size, state_dim, action_dim)
    runner = Runner(sess, actor, critic, buf, batch_size, discount, save_path)
    return runner
