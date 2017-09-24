'''
states: every single small grid
action: up, down, left, right(unless blocked by the wall)
'''

import numpy as np

# this shows how big the grid world is
a = np.zeros((4, 4))


# get the value function of adjacent grid(if exists)
def get_grid_p(grid, x, y):
    list1 = []
    list_direct = [0, 0, 0, 0]
    if x-1 > -1:
        list1.append(grid[x-1][y])
        list_direct[0] = 1
    if x+1 < 4:
        list1.append(grid[x+1][y])
        list_direct[1] = 1
    if y - 1 > -1:
        list1.append(grid[x][y-1])
        list_direct[2] = 1
    if y+1 < 4:
        list1.append(grid[x][y+1])
        list_direct[3] = 1
    return list1

# use b to update a during an iterations process
b = np.zeros((4, 4))

# main part: 8 value iterations
for i in range(0, 10):
    # search through the grid world
    for i in range(0, 4):
        for j in range(0, 4):
            list2 = get_grid_p(a, i, j)
            # find the average value of adjacent grids
            b[i][j] = round(-1+sum(list2)/len(list2),2)
    # the top left and bottom right corners are terminals
    b[0][0] = 0
    b[3][3] = 0
    print(b)
    print('------------------------')
    a = b
    b = np.zeros((4, 4))


