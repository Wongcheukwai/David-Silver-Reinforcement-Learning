'''
states: every single small grid
action: up, down, left, right(unless blocked by the wall)
'''

import numpy as np

# this shows how big the grid world is
a = np.zeros((4, 4))


# get the value function of adjacent grid(if exists)
def get_grid_v(grid, x, y):
    list1 = []
    if x-1 > -1:
        list1.append(grid[x-1][y])
    if x+1 < 4:
        list1.append(grid[x+1][y])
    if y - 1 > -1:
        list1.append(grid[x][y-1])
    if y+1 < 4:
        list1.append(grid[x][y+1])
    return list1

# use b to update a during an iterations process
b = np.zeros((4, 4))

# main part: 8 value iterations
for i in range(0, 8):
    # search through the grid world
    for i in range(0, 4):
        for j in range(0, 4):
            list2 = get_grid_v(a, i, j)
            # find the largest value of adjacent grids
            b[i][j] = max(list2)-1
    # the top left corner is the terminal
    b[0][0] = 0
    print(b)
    a = b
    b = np.zeros((4, 4))
    print('------------------')
