'''
the most original Monte-Carlo prediction for random walk
7 states, 
in which leftmost is the terminal with 0 reward and rightmost with 1 
two actions: move left or move right
V(s) = V(s) + ALPHA8* (reward + V(s+1) - V(s))
'''

import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.1

# to store V value for every state which is initialized by 5
line = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])

# use TD(0)


def TD (iteration):
    # start from the middle position
    n = 3
    for episodes in range(iteration):
        is_terminal = False
        while not is_terminal:
            c = np.random.uniform()
            # 50% to choose left
            if c > 0.5:
                b = n - 1
                # leftmost terminal state, episode ends
                if b == 0:
                    is_terminal = True
                    line[n] = line[n] + ALPHA*(line[b]-line[n])
                # not terminal, transfer to next state
                else:
                    is_terminal = False
                    line[n] = line[n] + ALPHA * (line[b] - line[n])
                    n = n-1
            # 50% to choose right
            else:
                # rightmost terminal state, episode ends, r = 1
                b = n + 1
                if b == 6:
                    is_terminal = True
                    line[n] = line[n] + ALPHA*(line[b]+1-line[n])
                # not terminal, transfer to next state
                else:
                    is_terminal = False
                    line[n] = line[n] + ALPHA * (line[b] - line[n])
                    n = n + 1
    return line[1:6]
l1 = TD(1)
l2 = TD(10)
l3 = TD(100)

# x axis: States, y axis: V Value
names = ['A','B','C','D','E']
x = range(len(names))
y_0 = [0.5, 0.5, 0.5, 0.5, 0.5]
y_1 = [1/6, 2/6, 3/6, 4/6, 5/6]
y_2 = l1
y_3 = l2
y_4 = l3

line_1 = plt.plot(x, y_0, 'ro-', label='Initial Value')
line_2 = plt.plot(x, y_1, 'bo-', label='True Value')
line_3 = plt.plot(x, y_2, 'mp-')
line_4 = plt.plot(x, y_3, 'g+-')
plt.plot(x, y_4, 'ko-', label='100 Episodes')
#plt.legend((line_1, line_2, line_4), ('Initial Value', 'True Value', '100 Episodes')

plt.xlabel('State')
plt.ylabel('Estimated value')

plt.xticks(x, names, rotation=45)
plt.margins(0.08)
plt.subplots_adjust(bottom=0.15)
plt.show()

