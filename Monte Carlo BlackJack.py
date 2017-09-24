import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# present the states
index_list = []
for i in range(1, 11):
    for j in range(12, 22):
        index_list.append(i*100+j)

# create an DataFrame to store Q(s,a)
q_table = pd.DataFrame(np.zeros((100, 2)),
                       index=index_list,
                       columns=['HIT', 'STICK'])

# create an DataFrame to store Returns(s,a)
return_table = pd.DataFrame(np.zeros((100, 2)),
                            index=index_list,
                            columns=['HIT', 'STICK'])

# create an DataFrame to store counter(s,a)
counter_table = pd.DataFrame(np.zeros((100, 2)),
                             index=index_list,
                             columns=['HIT', 'STICK'])

# create an DataFrame to store phi(s)
player_policy_table = pd.DataFrame(np.zeros((100, 1)),
                                   index=index_list,
                                   columns=['POLICY'])

for i in range(1, 11):
    for j in range(12, 22):
        player_policy_table.ix[i*100+j, :] = 'HIT'
for i in range(1, 11):
    for j in range(20, 22):
        player_policy_table.ix[i*100+j, :] = 'STICK'

# create an DataFrame to store deal's policy(sum >= 17 sticks, and hits otherwise)
dealer_policy_table = pd.DataFrame(np.zeros((21, 1)),
                                   index=range(1, 22),
                                   columns=['DEALER POLICY'])
for i in range(1, 18):
    dealer_policy_table.ix[i, :] = 'HIT'
for i in range(17, 22):
    dealer_policy_table.ix[i, :] = 'STICK'
#print(dealer_policy_table)


# use this function to draw card randomly (face cards count 10)
def draw_card():
    a = random.randint(1, 13)
    if a <= 9:
        a = a
    else:
        a = 10
    return a

for i in range(500000):
    # get the first state
    deal_card = random.randint(1, 10)
    print('deal_card', deal_card)
    copy_deal_card = deal_card  # to locate player policy index in case deal_card changed
    player_sum = random.randint(12, 21)
    print('player_sum', player_sum)
    list_state = []
    list_action = []
    # action given by player's policy
    action = player_policy_table.ix[deal_card*100+player_sum, 'POLICY']
    print('first action', action)
    # use a list to record state
    list_state.append(deal_card * 100 + player_sum)
    # use a list to record action
    list_action.append(action)
    # player sticks because of natural
    natural = 0
    if player_sum == 21:
        print('natural')
        natural = 1
        if deal_card == 1:
            new_card_111 = draw_card()
            print('new_card_111',new_card_111)
            if new_card_111 == 10:
                game_result = 'draw'
            else:
                game_result = 'win'
        elif deal_card == 10:
            new_card_1111 = draw_card()
            print('new_card_1111', new_card_1111)
            if new_card_1111 == 1:
                game_result = 'draw'
            else:
                game_result = 'win'
        else:
            game_result = 'win'
    # player hits
    if action == 'HIT':
        while action == 'HIT':
            print('at first we hit')
            # record the counter of Q(s,a)
            # draw a new card
            next_card = draw_card()
            print('next_card', next_card)
            player_sum += next_card
            print('here1,player sum', player_sum)
            # player loses the game
            if player_sum > 21:
                game_result = 'lost'
                break
            # action could be stick or hit
            else:
                action = player_policy_table.ix[deal_card * 100 + player_sum, 'POLICY']
                list_state.append(deal_card * 100 + player_sum)
                list_action.append(action)
                #counter_table.ix[deal_card * 100 + player_sum, action] += 1
    # player sticks, dealer's turn
    if action == 'STICK' and natural == 0 :
        print('----dealers turn')
        # dealer draws a card
        dealer_next_card = draw_card()
        print('dealer_next_card', dealer_next_card)
        deal_card += dealer_next_card
        print('dealer_card', deal_card)
        if deal_card > player_sum:
            game_result = 'lost'
        else:
            # dealer may hit or stick
            action_dealer = dealer_policy_table.ix[deal_card, 'DEALER POLICY']
            print('actions here2--------chose action', action_dealer)
            # dealer sticks
            if action_dealer == 'STICK':
                #print('here3--------stick')
                if deal_card < player_sum:
                    game_result = 'win'
                if deal_card == player_sum:
                    game_result = 'draw'
            # dealer hits
            while action_dealer == 'HIT':
                print('here4--------hit')
                dealer_next_card_1 = draw_card()
                print('here5--------newcard', dealer_next_card_1)
                deal_card += dealer_next_card_1
                print('here6--------newsum', deal_card)
                # dealer loses the game
                if deal_card > 21:
                    game_result = 'win'
                    break
                # action could be stick or hit
                else:
                    action_dealer = dealer_policy_table.ix[deal_card, 'DEALER POLICY']
                    print('here7-------',action_dealer)
    # both of dealer and player did not bust: Compare
    if deal_card > player_sum and natural == 0 and player_sum < 22 and deal_card < 22:
        game_result = 'lost'
    if deal_card == player_sum and natural == 0 and player_sum < 22 and deal_card < 22:
        game_result = 'draw'
    if deal_card < player_sum and natural == 0 and player_sum < 22 and deal_card < 22:
        game_result = 'win'
    print(game_result)

    # lets see my secret weapon to record the trajectory
    for k in range(len(list_state)):
        print('list_state', list_state[k])
        print('list_action', list_action[k])
        # update counter_table

        counter_table.ix[list_state[k], list_action[k]] += 1
        print('conter', list_state[k],list_action[k],counter_table.ix[list_state[k], list_action[k]])
        # update reward_table
        if game_result == 'win':
            return_table.ix[list_state[k], list_action[k]] += 1
        elif game_result == 'lost':
            return_table.ix[list_state[k], list_action[k]] -= 1
        else:
            return_table.ix[list_state[k], list_action[k]] += 0
        # update q_table
        q_table.ix[list_state[k], list_action[k]] = return_table.ix[list_state[k], list_action[k]]/counter_table.ix[list_state[k],list_action[k]]
        # update policy
        # reindex randomly in case that the first action of two equal maximum is always chosen
        choose_action = q_table.ix[list_state[k], :]
        choose_action = choose_action.reindex(np.random.permutation(choose_action.index))
        print('choose action max', choose_action.argmax())
        player_policy_table.ix[list_state[k], 'POLICY'] = choose_action.argmax()
        #print(player_policy_table)

x_axis_hit = []
y_axis_hit = []
x_axis_stick = []
y_axis_stick = []

'''
for dc in range(1, 11):
    for ps in range(12, 22):
        if player_policy_table.ix[dc*100+ps,'POLICY'] == 'HIT':
            x_axis_hit.append(dc)
            y_axis_hit.append(ps)
        if player_policy_table.ix[dc*100+ps,'POLICY'] == 'STICK':
            x_axis_stick.append(dc)
            y_axis_stick.append(ps)
plt.xlim(xmax=11,xmin=0)
plt.ylim(ymax=22,ymin=11)
plt.plot(x_axis_hit,y_axis_hit,'ro')
plt.plot(x_axis_stick,y_axis_stick,'bo')
plt.show()
'''

v_value = [[0 for x in range(10)] for y in range(10)]
for dc in range(10):
    for ps in range(10):
        v_value[dc][ps] = (q_table.ix[(dc+1)*100+ps+12, 'HIT'] * counter_table.ix[(dc+1)*100+ps+12, 'HIT'] + q_table.ix[(dc+1)*100+ps+12, 'STICK'] * counter_table.ix[(dc+1)*100+ps+12, 'STICK'])/(counter_table.ix[(dc+1)*100+ps+12, 'HIT'] + counter_table.ix[(dc+1)*100+ps+12, 'STICK'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X = []
for it in range(10):
    X.append(seq)

Y = np.asarray([[1 for x in range(10)] for y in range(10)])
for it in range(10):
    Y[it] = Y[it]*(it+12)

Z = np.asarray(v_value)
ax.plot_surface(X, Y, np.transpose(Z))

plt.show()






