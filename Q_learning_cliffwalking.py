'''
cliff walking problem solved by Q learning
Example 6.6 in P 140 of Reinforcement Learning: An Introduction
'''
import numpy as np
import pandas as pd

# initialization
row = 4
column = 12
ACTIONS = ['up', 'down', 'left', 'right']
ALPHA = 0.1

# a DataFrame to store Q value for every state, initialized to 0
# where index indicates the state number, column represent available actions
q_table = pd.DataFrame(
            np.zeros((row*column,len(ACTIONS))),
            index=range(row*column),
            columns=ACTIONS,
          )

# set nan to actions that can not be chosen(blocked by the wall) at each state
for i in range(column):
    q_table.ix[i]['up'] = np.nan
    q_table.ix[(row-1)*column+i]['down'] = np.nan
for i in range(row):
    q_table.ix[column*i]['left'] = np.nan
    q_table.ix[column * (i+1)-1]['right'] = np.nan


# functions used to choose actions
def pick_action(state):
    # if the agent falls into the cliff, it returns to the start state
    if (state >= 37) and (state <= 46):
        action = 'back to 36'
    # other wise it use epsilon-greedy policy to choose action
    else:
        c = np.random.uniform()
        # choose the action with largest Q value
        if c > 0.1:
            state_actions = q_table.ix[state,:]
            # change the position of index and elements in case there are 2 equal maximum
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            action = state_actions.argmax()
        # choose the action randomly
        else:
            state_actions = q_table.ix[state, :]
            state_actions.dropna(axis=0, inplace=True)
            action = np.random.choice(state_actions.index)
        print(action)
        return action


# environment returns agent reward and the next state
# give action in current state
def env_feedback(state, action):
    if action == 'back to 36':
        next_state = 36
        r = -100
    if action == 'up':
        next_state = state - column
        r = -1
    if action == 'down':
        next_state = state + column
        r = -1
    if action == 'left':
        next_state = state - 1
        r = -1
    if action == 'right':
        next_state = state + 1
        r = -1
    return next_state,r


# Calculate average reward of each episode
def average_r():
    sum_reward_list = []
    # start reinforcement learning using Q learning
    for i in range(500):
        # start from the bottom left
        sum_reward_1 = 0
        initial_state = 36
        is_terminal = False
        while not is_terminal:
            # pick action for next state
            action_1 = pick_action(initial_state)
            # environment feed back reward and next state
            next_state_1, r_1 = env_feedback(initial_state, action_1)
            sum_reward_1 += r_1
            # when the agent in normal states
            if initial_state < 37:
                q_target = r_1 + q_table.iloc[next_state_1, :].max()
                q_predict = q_table.ix[initial_state, action_1]
                q_table.ix[initial_state, action_1] += ALPHA * (q_target - q_predict)
                is_terminal = False
                initial_state = next_state_1
            # when the agent falls into the cliff
            if (initial_state >= 37) and (initial_state <= 46):
                q_target = r_1 + q_table.iloc[next_state_1, :].max()
                print(q_target)
                q_predict = q_table.ix[initial_state, 'up']
                q_table.ix[initial_state, :] += ALPHA * (q_target - q_predict)
                is_terminal = False
                initial_state = 36
            # when the agent reaches the goal(terminal state)
            if initial_state == 47:
                is_terminal = True
        sum_reward_list.append(sum_reward_1)
        print('-------')
    return sum_reward_list

sum_reward_list_1 = average_r()
sum_reward_list_2 = average_r()
sum_reward_list_3 = average_r()
sum_reward_list_all = [0]*500
for i in range(500):
    sum_reward_list_all[i] = round((sum_reward_list_1[i] + sum_reward_list_2[i] + sum_reward_list_3[i])/3,2)

print(sum_reward_list_1)
print(q_table)

