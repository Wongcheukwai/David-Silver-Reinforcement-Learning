from tile_coding import tiles, IHT
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# size of a list
maxSize = 2048*2
# the list to store indices of tile coding
iht = IHT(maxSize)
# parameter theta
weights = [0]*maxSize
# number of tilings
numTilings = 8
# alpha: learning rate or step size
stepSize = 0.1/numTilings

EPSILON = 0.9
GAMMA = 0.9

# bound of position
position_max, position_min = 0.5, -1.2
# bound of velocity
velocity_max, velocity_min = 0.07, -0.07

# plot parameters
position_interval = 0.1
velocity_interval = 0.01
position_num = 1+int(((position_max-position_min)/position_interval))
velocity_num = 1+int(((velocity_max-velocity_min)/velocity_interval))


# use tile coding to get the indices of activated features(tiles)
# x represents position, y velocity, action 1(forward),0(zero)and -1(reverse)
def get_tile(x, y, action=[]):
    return tiles(iht, numTilings, [numTilings*x/(position_max-position_min), numTilings*y/(velocity_max-velocity_min)], action)


# use tile list to estimate Q(s,a,theta)
def get_q_value(x, y, action=[]):
    my_tiles_0 = get_tile(x, y, action)
    estimate = 0
    for tile_0 in my_tiles_0:
        estimate += weights[tile_0]  # since all activated features equal to 1
    return estimate


# to update theta
def update(x, y, z, action=[]):
    my_tales_1 = get_tile(x, y, action)
    for tile_1 in my_tales_1:
        weights[tile_1] += stepSize * z  # learn weights


# create a list to choose action
def get_q_list(x, y):
    q_list = []
    full_throttle_forward = get_q_value(x, y, [1])
    q_list.append(full_throttle_forward)
    zero_throttle = get_q_value(x, y, [0])
    q_list.append(zero_throttle)
    full_throttle_reverse = get_q_value(x, y, [-1])
    q_list.append(full_throttle_reverse)
    return q_list


# use epsilon greedy to choose actions
def epsilon_greedy(q_list):
    # act non-greedy or state-action have no value
    if np.random.uniform() > EPSILON:
        print('random')
        index_action = q_list.index(np.random.choice(q_list))
    # act greedy
    else:
        print('max')
        max_action_value = max(q_list)
        print('max_action_value', max_action_value)
        index_action = q_list.index(max_action_value)
        print('index_action', index_action)
    action_list = []
    # q_list[0] stores the action value for forward, [1] zero, [2] reverse
    if index_action == 0:
        action_list = [1]

    elif index_action == 1:
        action_list = [0]
    else:
        action_list = [-1]
    print('action_list',action_list)
    return action_list

episode = 0
for i in range(100):
    # initial position -0.6=<x=<-0.4

    episode += 1

    print('episode', episode)
    ini_position = random.uniform(-0.6, -0.4)

    while ini_position == -0.4:
        ini_position = random.uniform(-0.6, -0.4)

    # initial velocity v=0
    ini_velocity = 0

    # get the initial q list which contained q value for three actions
    q_list_0 = get_q_list(ini_position, ini_velocity)
    print('qlist0',q_list_0)
    # use epsilon_greedy method to choose first action
    first_action = epsilon_greedy(q_list_0)
    print('first action',first_action)
    position_current = ini_position
    velocity_current = ini_velocity
    action_current = first_action
    is_terminal = False
    step = 0
    while not is_terminal:
        step += 1
        print('step',step)
        r = -1
        print('velocity_current',velocity_current)
        print('position_current',position_current)
        # calculate the next position and velocity
        print('action_current',action_current[0])
        velocity_next = velocity_current + 0.001 * action_current[0] - 0.0025 * math.cos(3*position_current)
        print('velocity_next',velocity_next)
        position_next = position_current + velocity_next
        print('position_next',position_next)
        # set bound for velocity
        if velocity_next <= velocity_min:
            velocity_next = velocity_min
        if velocity_next >= velocity_max:
            velocity_next = velocity_max
        # deal with the bound for position
        if position_next <= position_min:
            position_next = position_min
            velocity_next = 0
        #  the agent reaches the goal
        if position_next >= position_max:
            position_next = position_max
            is_terminal = True
            q_value_0 = get_q_value(position_current, velocity_current, action_current)
            error_0 = (r - q_value_0) * numTilings
            update(position_current, velocity_current, error_0, action_current)
        # normal position
        if (position_next >= position_min) and (position_next < position_max):
            q_list_1 = get_q_list(position_next, velocity_next)
            action_next = epsilon_greedy(q_list_1)
            is_terminal = False
            q_value_1 = get_q_value(position_current, velocity_current, action_current)
            q_value_2 = get_q_value(position_next, velocity_next, action_next)
            error_1 = (r + GAMMA * q_value_2 - q_value_1) * numTilings
            update(position_current, velocity_current, error_1, action_current)
            position_current = position_next
            velocity_current = velocity_next
            action_current = action_next

# this matrix stores the minus max of q over three actions for every position velocity comb
max_q_matrix = np.zeros((position_num, velocity_num))

for it in range(max_q_matrix.shape[0]):
    for it2 in range(max_q_matrix.shape[1]):
        max_q_matrix[it, it2] = -max(get_q_value(position_min + it*position_interval,
                                                 velocity_min + it2*velocity_interval,
                                                 [1]),
                                     get_q_value(position_min + it * position_interval,
                                                 velocity_min + it2 * velocity_interval,
                                                 [0]),
                                     get_q_value(position_min+it*position_interval,
                                                 velocity_min+it2*velocity_interval,
                                                 [-1])
                                     )

print(max_q_matrix)


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

X = np.linspace(position_min, position_max, num=position_num,endpoint=True)
Y = np.linspace(velocity_min, velocity_max, num=velocity_num,endpoint=True)
X, Y = np.meshgrid(X, Y)
Z = np.transpose(max_q_matrix)

plt.title('-max q over a')
plt.xlabel('position')
plt.ylabel('velocity')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()


