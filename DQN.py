import random
import numpy as np
import math
import gym
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# parameters for epsilon greedy
EPSILON = 0.9
# reward decay
GAMMA = 0.9

# hidden layer
hidden_layer = 20
l_r = 0.01
# memory of experience replay
memory_size = 500
memory_column = 5
mini_batch = 32
step = 0
learning_inter = 200

# create two NN using sckitlearn
old_nn = MLPClassifier( hidden_layer_sizes=hidden_layer, learning_rate_init=l_r, batch_size=mini_batch)
new_nn = MLPClassifier( hidden_layer_sizes=hidden_layer, learning_rate_init=l_r, batch_size=mini_batch)
x = np.zeros([mini_batch, 3])
y = np.zeros([mini_batch, 1])
old_nn.fit(x, y)
new_nn.fit(x, y)

# import the environment
env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


def choose_action(position, velocity):
    if np.random.uniform() < EPSILON:

        q_list = [new_nn.predict(np.asarray([position, velocity, 0]).reshape(1, -1)),
                  new_nn.predict(np.asarray([position, velocity, 1]).reshape(1, -1)),
                  new_nn.predict(np.asarray([position, velocity, 2]).reshape(1, -1))]
        action_chosen = np.argmax(q_list)
    else:
        action_chosen = env.action_space.sample()
    return action_chosen


# experience replay
memory = np.zeros((memory_size, memory_column))
counter = 0
while counter < memory_size:
    observation = env.reset()
    while counter < memory_size:
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        memory[counter, 0], memory[counter, 1] = observation[0], observation[1]
        memory[counter, 2], memory[counter, 3], memory[counter, 4] = action, observation_[0], observation_[1]
        counter += 1
        if observation_[0] >= env.observation_space.high[0]:
            break
        observation = observation_

for i_episode in range(10):
    observation = env.reset()
    print(observation)
    is_terminal = False
    index = 0
    while not is_terminal:
        env.render()
        action_0 = choose_action(observation[0], observation[1])
        observation_1, r_, done, info = env.step(action_0)
        memory[index % memory_size, 0], memory[index % memory_size, 1] = observation[0], observation[1]
        memory[index % memory_size, 2], memory[index % memory_size, 3], memory[index % memory_size, 4] = action_0, observation_1[0], observation_1[1]
        index += 1
        observation = observation_1

        # choose minimum batch
        batch_index = np.random.choice(memory_size, size=mini_batch)
        batch_memory = memory[batch_index, :]
        e_r_l_new = batch_memory[:, 0:3]
        e_r_l_old = batch_memory[:, 3:5]
        q_target = np.zeros((mini_batch, 1))
        counter_2 = 0
        for element in e_r_l_old:
            q_target[counter_2] = r_ + GAMMA * max(old_nn.predict([np.append(element, 0)]), \
                                                   old_nn.predict([np.append(element, 1)]), \
                                                   old_nn.predict([np.append(element, 2)]))
            counter_2 += 1
        q_target = q_target.ravel()
        new_nn.fit(e_r_l_new, q_target)

        if step % learning_inter == 0:
            old_nn.coefs_ = new_nn.coefs_
            old_nn.intercepts_ = new_nn.intercepts_
            # reach the right top
        if observation[0] == env.observation_space.high[0]:
            break

