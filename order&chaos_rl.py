"""
Reference implementation of the Tic-Tac-Toe value function learning agent described in Chapter 1 of
"Reinforcement Learning: An Introduction" by Sutton and Barto. The agent contains a lookup table that
maps states to values, where initial values are 1 for a win, 0 for a draw or loss, and 0.5 otherwise.
At every move, the agent chooses either the maximum-value move (greedy) or, with some probability
epsilon, a random move (exploratory); by default epsilon=0.1. The agent updates its value function
(the lookup table) after every greedy move, following the equation:
    V(s) <- V(s) + alpha * [ V(s') - V(s) ]
This particular implementation addresses the question posed in Exercise 1.1:
    What would happen if the RL agent taught itself via self-play?
The result is that the agent learns only how to maximize its own potential payoff, without consideration
for whether it is playing to a win or a draw. Even more to the point, the agent learns a myopic strategy
where it basically has a single path that it wants to take to reach a winning state. If the path is blocked
by the opponent, the values will then usually all become 0.5 and the player is effectively moving randomly.
Created by Wesley Tansey
1/21/2013
Edited to Order and Chaos by Shivaen Ramshetty
5/24/2020
Code released under the MIT license.
"""

import random
from copy import copy, deepcopy
import csv
import matplotlib.pyplot as plt

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3

BOARD_FORMAT = "--------------------------------------------------------\n| {0} | {1} | {2} | {3} | {4} | {5} |\n|------------------------------------------------------\n| {6} | {7} | {8} | {9} | {10} | {11} |\n|------------------------------------------------------\n| {12} | {13} | {14} | {15} | {16} | {17} |\n|--------------------------------------------------------\n| {18} | {19} | {20} | {21} | {22} | {23} |\n|--------------------------------------------------------\n| {24} | {25} | {26} | {27} | {28} | {29} |\n|--------------------------------------------------------\n| {30} | {31} | {32} | {33} | {34} | {35} |\n|--------------------------------------------------------"
NAMES = [' ', 'X', 'O']
def printboard(state):
    cells = []
    for i in range(6):
        for j in range(6):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))

def emptystate():
    return [[EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY],[EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY],[EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY],[EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY],[EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY],[EMPTY,EMPTY,EMPTY,EMPTY,EMPTY,EMPTY]]

def gameover(state):
    for i in range(6):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2] and state[i][0] == state[i][3] and state[i][0] == state[i][4]:
            return state[i][0]
        if state[i][1] != EMPTY and state[i][1] == state[i][2] and state[i][1] == state[i][3] and state[i][1] == state[i][4] and state[i][1] == state[i][5]:
            return state[i][1]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i] and state[0][i] == state[3][i] and state[0][i] == state[4][i]:
            return state[0][i]
        if state[1][i] != EMPTY and state[1][i] == state[2][i] and state[1][i] == state[3][i] and state[1][i] == state[4][i] and state[1][i] == state[5][i]:
            return state[1][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2] and state[0][0] == state[3][3] and state[0][0] == state[4][4]:
        return state[0][0]
    if state[1][1] != EMPTY and state[1][1] == state[2][2] and state[1][1] == state[3][3] and state[1][1] == state[4][4] and state[1][1] == state[5][5]:
        return state[1][1]
    if state[1][0] != EMPTY and state[1][0] == state[2][1] and state[1][0] == state[3][2] and state[1][0] == state[4][3] and state[1][0] == state[5][4]:
        return state[1][0]
    if state[0][1] != EMPTY and state[0][1] == state[1][2] and state[0][1] == state[2][3] and state[0][1] == state[3][4] and state[0][1] == state[4][5]:
        return state[0][1]
    if state[0][5] != EMPTY and state[0][5] == state[1][4] and state[0][5] == state[2][3] and state[0][5] == state[3][2] and state[0][5] == state[4][1]:
        return state[0][5]
    if state[1][4] != EMPTY and state[1][4] == state[2][3] and state[1][4] == state[3][2] and state[1][4] == state[4][1] and state[1][4] == state[5][0]:
        return state[1][4]
    if state[0][4] != EMPTY and state[0][4] == state[1][3] and state[0][4] == state[2][2] and state[0][4] == state[3][1] and state[0][4] == state[4][0]:
        return state[0][4]
    if state[1][5] != EMPTY and state[1][5] == state[2][4] and state[1][5] == state[3][3] and state[1][5] == state[4][2] and state[1][5] == state[5][1]:
        return state[1][5]
    for i in range(6):
        for j in range(6):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW

def last_to_act(state):
    countx = 0
    counto = 0
    for i in range(6):
        for j in range(6):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    if idx > 35:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = int(idx / 6)
        j = int(idx % 6)
        for val in range(6):
            state[i][j] = val
            enumstates(state, idx+1, agent)

class Agent(object):
    def __init__(self, player, verbose = False, lossval = 0, learning = True):
        self.values = {}
        self.player = player
        self.verbose = verbose
        self.lossval = lossval
        self.learning = learning
        self.epsilon = 0.1
        self.alpha = 0.99
        self.prevstate = None
        self.prevscore = 0
        self.count = 0
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY
        return move

    def random(self, state):
        available = []
        for i in range(6):
            for j in range(6):
                if state[i][j] == EMPTY:
                    available.append((i,j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(6):
            for j in range(6):
                if state[i][j] == EMPTY:
                    state[i][j] = self.player
                    val = self.lookup(state)
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxmove = (i, j)
                    if self.verbose:
                        cells.append('{0:.3f}'.format(val).center(6))
                elif self.verbose:
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)

    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner)

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def printvalues(self):
        vals = deepcopy(self.values)
        for key in vals:
            print(key)
            state = [list(key[0]),list(key[1]),list(key[2]), list(key[3]), list(key[4]), list(key[5])]
            cells = []
            for i in range(6):
                for j in range(6):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        return (tuple(state[0]),tuple(state[1]),tuple(state[2]), tuple(state[3]), tuple(state[4]), tuple(state[5]))

    def log(self, s):
        if self.verbose:
            print(s)

class Human(object):
    def __init__(self, player):
        self.player = player

    def action(self, state):
        printboard(state)
        action = input('Your move? i.e. x,y : ')
        return (int(action.split(',')[0]),int(action.split(',')[1]))

    def episode_over(self, winner):
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))

def play(agent1, agent2):
    state = emptystate()
    for i in range(36):
        if i % 2 == 0:
            move = agent1.action(state)
        else:
            move = agent2.action(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner

def measure_performance_vs_random(agent1, agent2):
    epsilon1 = agent1.epsilon
    epsilon2 = agent2.epsilon
    agent1.epsilon = 0
    agent2.epsilon = 0
    agent1.learning = False
    agent2.learning = False
    r1 = Agent(1)
    r2 = Agent(2)
    r1.epsilon = 1
    r2.epsilon = 1
    probs = [0,0,0,0,0,0]
    games = 100
    for i in range(games):
        winner = play(agent1, r2)
        if winner == PLAYER_X:
            probs[0] += 1.0 / games
        elif winner == PLAYER_O:
            probs[1] += 1.0 / games
        else:
            probs[2] += 1.0 / games
    for i in range(games):
        winner = play(r1, agent2)
        if winner == PLAYER_O:
            probs[3] += 1.0 / games
        elif winner == PLAYER_X:
            probs[4] += 1.0 / games
        else:
            probs[5] += 1.0 / games
    agent1.epsilon = epsilon1
    agent2.epsilon = epsilon2
    agent1.learning = True
    agent2.learning = True
    return probs

def measure_performance_vs_each_other(agent1, agent2):
    #epsilon1 = agent1.epsilon
    #epsilon2 = agent2.epsilon
    #agent1.epsilon = 0
    #agent2.epsilon = 0
    #agent1.learning = False
    #agent2.learning = False
    probs = [0,0,0]
    games = 100
    for i in range(games):
        winner = play(agent1, agent2)
        if winner == PLAYER_X:
            probs[0] += 1.0 / games
        elif winner == PLAYER_O:
            probs[1] += 1.0 / games
        else:
            probs[2] += 1.0 / games
    #agent1.epsilon = epsilon1
    #agent2.epsilon = epsilon2
    #agent1.learning = True
    #agent2.learning = True
    return probs


if __name__ == "__main__":
    p1 = Agent(1, lossval = -1)
    p2 = Agent(2, lossval = -1)
    r1 = Agent(1, learning = False)
    r2 = Agent(2, learning = False)
    r1.epsilon = 1
    r2.epsilon = 1
    series = ['P1-Win','P1-Lose','P1-Draw','P2-Win','P2-Lose','P2-Draw']
    #series = ['P1-Win', 'P2-Win', 'Draw']
    colors = ['r','b','g','c','m','b']
    markers = ['+', '.', 'o', '*', '^', 's']
    f = open('results.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(series)
    perf = [[] for _ in range(len(series) + 1)]
    for i in range(10000):
        if i % 10 == 0:
            print('Game: {0}'.format(i))
            probs = measure_performance_vs_random(p1, p2)
            writer.writerow(probs)
            f.flush()
            perf[0].append(i)
            for idx,x in enumerate(probs):
                perf[idx+1].append(x)
        winner = play(p1,p2)
        p1.episode_over(winner)
        #winner = play(r1,p2)
        p2.episode_over(winner)
    f.close()
    for i in range(1,len(perf)):
        plt.plot(perf[0], perf[i], label=series[i-1], color=colors[i-1])
    plt.xlabel('Episodes')
    plt.ylabel('Probability')
    plt.title('RL Agent Performance vs. Random Agent\n({0} loss value, self-play)'.format(p1.lossval))
    #plt.title('P1 Loss={0} vs. P2 Loss={1}'.format(p1.lossval, p2.lossval))
    plt.legend()
    plt.show()
    #plt.savefig('p1loss{0}vsp2loss{1}.png'.format(p1.lossval, p2.lossval))
    plt.savefig('selfplay_random_{0}loss.png'.format(p1.lossval))
    while True:
        p2.verbose = True
        p1 = Human(1)
        winner = play(p1,p2)
        p1.episode_over(winner)
        p2.episode_over(winner)