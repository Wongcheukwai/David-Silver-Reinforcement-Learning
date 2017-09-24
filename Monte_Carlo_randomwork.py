'''
the most original Monte-Carlo prediction for random walk
7 states, 
in which leftmost is the terminal with 0 reward and rightmost with 1 
two actions: move left or move right
for every state s,
N(s) ← N(s)+1
S（s）← S（s)+ Gt
V(s) = S(s)/N(s)
Calculate the RMS error between true vale and estimated value
'''

import numpy as np
import pandas as pd
import math
from scipy.interpolate import spline
import matplotlib.pyplot as plt

list1 = [0, 0, 0, 0, 0, 0] # N(s) counter
list2 = [0, 0, 0, 0, 0, 0] # Calculate Gt
list3 = [0, 0, 0, 0, 0, 0] # V(s) = S(s)/N(s)
list_100 = [] # to store the EMS error every episode

# initialize RMS for every state
RMS_1 = 0
RMS_2 = 0
RMS_3 = 0
RMS_4 = 0
RMS_5 = 0
RMS_F = 0
# calculate RMS error from Episode 10 to 100
for episodes in range(10, 101):
    # start from the middle position
    n = 3
    # create q_table to generate episodes
    q_table = pd.DataFrame(columns=['States', 'Reward'])
    q_table = q_table.append(
        pd.Series({'States': n, 'Reward': 0}),
        ignore_index=True
    )
    is_terminal = False
    while not is_terminal:
        c = np.random.uniform()
        # 50% to choose left
        if c > 0.5:
            b = n - 1
            # leftmost terminal state, episode ends
            if b == 0:
                is_terminal = True
                r = 0
                q_table = q_table.append(
                    pd.Series({'States': 'terminal_0', 'Reward': r}),
                    ignore_index=True
                )
            # not terminal, transfer to next state
            else:
                is_terminal = False
                r = 0
                q_table = q_table.append(
                    pd.Series({'States': b, 'Reward': r}),
                    ignore_index=True
                )
                n = n - 1
        # 50% to choose right
        else:
            # rightmost terminal state, episode ends, r = 1
            b = n + 1
            if b == 6:
                is_terminal = True
                r = 1
                q_table = q_table.append(
                    pd.Series({'States': 'terminal_1', 'Reward': r}),
                    ignore_index=True
                )
            # not terminal, transfer to next state
            else:
                is_terminal = False
                r = 0
                q_table = q_table.append(
                    pd.Series({'States': b, 'Reward': r}),
                    ignore_index=True
                )
                n = n + 1
    # delete the '#' and you can see the sequence and reward of every episode
    # print(q_table)

    # some extra work to calculator the reward, V(s) and EMS error
    a = q_table.ix[:, 'States'].tolist()
    a.pop()
    c = list(set(a))

    for i in c:
        list2[i] += 1

    b = q_table.ix[:, 'Reward'].tolist()
    for i in c:
        first_index = a.index(i)
        for j in range(first_index,len(b)):
            list1[i] += b[j]

    for q in range(1, 6):
        if list2[q] == 0:
            list3[q] = np.square(0 - q/6)
        else:
            list3[q] = np.square(list1[q] / list2[q] - q / 6)

    RMS_1 += list3[1]
    RMS_1 += list3[2]
    RMS_1 += list3[3]
    RMS_1 += list3[4]
    RMS_1 += list3[5]

    RMS_1 = math.sqrt(RMS_1/episodes)
    RMS_2 = math.sqrt(RMS_2/episodes)
    RMS_3 = math.sqrt(RMS_3/episodes)
    RMS_4 = math.sqrt(RMS_4/episodes)
    RMS_5 = math.sqrt(RMS_5/episodes)

    RMS_F = (RMS_1 + RMS_2 + RMS_3 + RMS_4 + RMS_5)/5
    list_100.append(RMS_F)

print(len(list_100))
x_axis = np.arange(10, 101)

# x axis: episode, y axis: empirical RMS error, averaged over states
xnew = np.linspace(x_axis.min(), x_axis.max(), 300)  # 300 represents number of points to make between T.min and T.max
plt.title('Monte Carlo')
power_smooth = spline(x_axis, list_100, xnew)
plt.xlabel('Walks / Episodes')
plt.ylabel('RMS error')
plt.plot(xnew, power_smooth)
plt.show()



