
import numpy as ql
# R is The Reward Matrix for each state
R = ql.matrix([ [0,0,0,0,1,0],
                [0,0,0,1,0,1],
                [0,0,100,1,0,0],
                [0,1,1,0,1,0],
                [1,0,0,1,0,0],
                [0,1,0,0,0,0] ])

# Q is the Learning Matrix in which rewards will be learned/stored
Q = ql.matrix(ql.zeros([6,6]))

# Gamma : 
# It's a form of penalty or uncertainty for learning
# If the value is 1 , the rewards would be too high.
# This way the system knows it is learning.
gamma = 0.8

# agent_s_state. The agent the name of the system calculating
# s is the state the agent is going from and s' the state it's going to
# this state can be random or it can be chosen as long as the rest of the choices
# are not determined. Randomness is part of this stochastic process
agent_s_state = 5

# The possible "a" actions when the agent is in a given state
def possible_actions(state):
    current_state_row = R[state,]
    possible_act = ql.asarray(current_state_row > 0).nonzero()
    return possible_act[1] # Return the index of posibility

# Get available actions in the current state
PossibleAction = possible_actions(agent_s_state)

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def ActionChoice(available_actions_range):
    if(sum(PossibleAction) > 0):
        next_action = int(ql.random.choice(PossibleAction,1))
    if(sum(PossibleAction) <= 0):
        next_action = int(ql.random.choice(5,1))
    return next_action

# Sample next action to be performed
action = ActionChoice(PossibleAction)
print('action',action)

# A version of Bellman's equation for reinforcement learning using the Q function
# This reinforcement algorithm is a memoryless process
# The transition function T from one state to another
# is not in the equation below.  T is done by the random choice above
# print(Q[action,])
def reward(current_state, action, gamma):
    print('bf_Q',Q)

    Max_State = ql.asarray(Q[action,] == ql.max(Q[action,])).nonzero()[1]
    print('bf_max_state',Max_State)
    if Max_State.shape[0] > 1: # Check if max_state have more than 1 column
        Max_State = ql.random.choice(Max_State, size=1)
        print('max_state', Max_State)
    else:
        Max_State = int(Max_State)
    print('af_max_state',Max_State)
    MaxValue = Q[action, Max_State]
    print('max_value', MaxValue)
    # Bellman's MDP based Q function
    Q[current_state, action] = R[current_state, action] + gamma * MaxValue

# Rewarding Q matrix
reward(agent_s_state, action, gamma)
print('reward',Q[agent_s_state,])
print('')
# Learning over n iterations depending on the convergence of the system
# A convergence function can replace the systematic repeating of the process
# by comparing the sum of the Q matrix to that of Q matrix n-1 in the
# previous episode
for i in range(25):
    current_state = ql.random.randint(0, int(Q.shape[0]))
    print('curr_state',current_state)
    PossibleAction = possible_actions(current_state)
    print('poss_act',PossibleAction)
    action = ActionChoice(PossibleAction)
    print('action',action)
    reward(current_state,action,gamma)
    print('reward',Q)
    print('')
    
# # Displaying Q before the norm of Q phase
# print("Q  :")
# print(Q)

# # # Norm of Q
# print("Normed Q :", Q)
# print(Q/ql.max(Q)*100)
