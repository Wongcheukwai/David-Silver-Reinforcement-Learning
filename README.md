# David Silver Reinforcement Learning

Main algorithms implementation in Pyhton

Hello there! I’m **Wong**

All these examples are chosen from David Silver's Reinforcement Leanrning class to demonstrate main algorithms in RL. 



## Value Iteration

**Environment** 

![](/Users/wongcheukwai/Desktop/1505801413419.jpg)

* 4*4 gridworld， the top left corner is the goal
* when the agent's path is blocked by the wall, it will not choose the action leading to that
* state is the grid the agent is in
* action : up, down, left, right(if not blocked by the wall)

**Result** 

* value function on each state(grid) 
* `Value Iteration Gridworld`


## Policy Iteration

**Environment** 

![](/Users/wongcheukwai/Desktop/policyiteration.jpg)

* 4*4 gridworld，the top left and bottom right corner is the goal
* when the agent's path is blocked by the wall, it will not choose the action leading to that
* state is the grid the agent is in
* action : up, down, left, right (if not blocked by the wall)
* reward is -1 until the agent reaches the goal

**Result** 

* value function on each state(grid)
* `Policy Iteration Gridworld`
* slightly different from David's result since he assumes that the agent can choose "hit the wall" action, after which the agent stays the same grid in the next state.



## Monte Carlo Prediton

**Environment** 

![](/Users/wongcheukwai/Desktop/rw1.jpg)

* start from state C, terminates on both sides
* action : left, right 
* reward is -1 when the agent terminates on the rightmost goal, 0 leftmost

**Result** 

* value function learned by Monte Carlo prediciton
* `Monte Carlo Predicition`


## TD(0) Prediton

**Environment** 

![](/Users/wongcheukwai/Desktop/rw1.jpg)

* start from state C, terminates on both sides
* action : left, right 
* reward is -1 when the agent terminates on the rightmost goal, 0 leftmost

**Result** 

* value function learned by TD(0) Prediton
* `TD(0) Prediton`


## Monte Carlo Control

**Environment** 

![]( David-Silver-Reinforcement-Learning/example_pics/bj.jpg )

* Black Jack
* game rule: 
 	1.  First step : both the dealer and player have two cards, one of the dealers is called "showing card"
	*  Second step : the player can choose "stick"(stop getting card) or "hit"(get one more card) until the player's sum is over 21 or choose to stick.
	*  Third step: the dealer's policy is fixed: if the dealer's sum is over 17, twist, otherwise,stick.
	*  The person whose sum is over 21 loses ( busted ).
	*  If no one busts, the person whose sum is bigger wins.
	*  Assumption: 1-9 stands for 1-9, 10-13 stands for 10, the probalibility of getting any card is the same for every round (get card with replacement).
	*  This is a non-unsable ace situation （A stands for 1）
* states: player's sum ( 12-21 ) and dealer's showing card ( 1-10 )
* action: hit, stick
* reward: win 1, draw 0, lose -1
	

**Result** 

* policy result (when to twist and when to result)
* `Average Value Function`


## Q-Learning and Sarsa
**Environment** 

![](/Users/wongcheukwai/Desktop/cw.jpg)

* Cliff Walking
* agent finds its way to the goal on the bottom right corner (terminal)
* If the agent reaches the cliff, it will be sent back to the start point(the bottom left corner)
* states: player's sum ( 12-21 ) and dealer's showing card ( 1-10 )
* action: up, down, left, right (if not blocked by the wall) 
* reward: -1 every step except reaching the cliff ( -100 )	

**Result** 

* `Sarsa & Qlearning Average Reward Comparison`
*  optimal path abtained by Q Learning
*  a much safer path abtained by Sarsa ( more iterations required )


## Sarsa Semi-Gradient Approximation
**Environment** 

![](/Users/wongcheukwai/Desktop/moc.jpg)

* Mountain Car
* agent（car） tried to reach the goal by moving back and forth in the valley ( even at full throttle the car cannot accelerate up the steep slope )
* states: contined state including postion and velocity, use `tile_coding' to represent continous states
* action: full throttle forward, zero throttle and full throttle reverse 
* reward: -1 every step until it reaches the goal	

**Result** 

* `-maxQ(s,a,theta) Mountain Car`



