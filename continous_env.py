import gym

from rl.ddpg_learner import DDPGLearner
from rl.models import QLearningKerasModel, SimpleDDPGModel
from rl.q_learner import QLearner
from rl.replay_buffer import ReplayBuffer
import tensorflow as tf

env = gym.make('Reacher-v1')

actions_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

buffer = ReplayBuffer(100000)

with tf.Session() as session:

    model = SimpleDDPGModel(state_dim, actions_dim, session, 0.001, 0.001)
    session.run(tf.initialize_all_variables())
    learner = DDPGLearner(buffer, model, 128, 0.99, state_dim, actions_dim)

    for i_episode in range(20000):
        s0 = env.reset()
        for t in range(1000):
            env.render()
            eps = 0.98
            scales = [[-1.0, 1.0], [-1.0, 1.0]]
            action = learner.choose_action(s0, eps)
            scaled_action = learner.scale_action(action, scales)
            #print(action)
            s1, reward, done, _ = env.step(scaled_action)
            learner.add(s0, action, reward, done, s1)
            learner.train()
            s0 = s1
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
