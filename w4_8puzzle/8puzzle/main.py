import numpy as np
from pprint import pprint
import heapq
import copy
from itertools import count
import time

def gen_goal_board(size=3):
    return np.array(list(range(1,size**2))+[0]).reshape((size, size))


def find_idx(arr, val):
    idx = np.nonzero(arr.flatten() == val)
    i, j = np.unravel_index(idx, arr.shape)
    return i[0][0], j[0][0]

def hamming_priority(state, goal_board):
    i, j = find_idx(state[0], 0)
    gi, gj = find_idx(goal_board, 0)
    zero_val = 0 if i==gi and j == gj else 1
    return state[1]+np.sum(state[0]!=goal_board)-zero_val

def manhattan_priority(state, goal_board):
    man_dist = 0
    for el_idx, el in enumerate(state[0].flatten()):
        if el != 0 :
            el_i, el_j = np.unravel_index(el_idx, goal_board.shape)
            goal_i, goal_j = find_idx(goal_board, el)
            man_dist += abs(goal_i-el_i) + abs(goal_j-el_j)
    return state[1] + man_dist


def neighbour_boards(board):
    i,j =  find_idx(board, 0)
    neighbours = []
    if i+1<=board.shape[0]-1:
        neighbour = np.copy(board)
        neighbour[i,j], neighbour[i+1,j] = neighbour[i+1,j], neighbour[i,j]
        neighbours.append(neighbour)
    if i-1 >= 0:
        neighbour = np.copy(board)
        neighbour[i, j], neighbour[i - 1, j] = neighbour[i - 1, j], neighbour[i, j]
        neighbours.append(neighbour)
    if j+1<=board.shape[1]-1:
        neighbour = np.copy(board)
        neighbour[i,j], neighbour[i,j+1] = neighbour[i,j+1], neighbour[i,j]
        neighbours.append(neighbour)
    if j-1 >= 0:
        neighbour = np.copy(board)
        neighbour[i, j], neighbour[i, j-1] = neighbour[i, j-1], neighbour[i, j]
        neighbours.append(neighbour)
    return neighbours

def neighbour_states(state):
    return [(n, state[1]+1, state) for n in neighbour_boards(state[0])]

def search(init_board, goal_board, pf=hamming_priority, timeout=5):
    # board, moves, prev_board
    init_state = (init_board, 0, None)
    q = []
    tiebreaker = count()
    n_states_enq = 0
    heapq.heappush(q, (pf(init_state,goal_board), next(tiebreaker), init_state))
    n_states_enq += 1
    max_moves = 1000
    max_it = 100000
    i = 0
    start = time.time()
    while True:
        if i%10000 ==0:
            end = time.time()
            if (end-start)/60. > timeout:
                return (init_state, end-start)
        _, _, state = heapq.heappop(q)
        i += 1
        #if i> max_it:
        #    return state, n_states_enq
        #print i
        #print
        #print state[0]
        if state[1] > max_moves:
            return state, n_states_enq
        if np.sum(state[0] != goal_board) == 0:
            return state, n_states_enq

        for s in neighbour_states(state):
            if state[2] is None or np.sum(state[2][0] != s[0]) > 0:
                heapq.heappush(q, (pf(s, goal_board), next(tiebreaker), s))
                n_states_enq += 1

def read_init(path):
    with open(path, 'r') as fin:
        idx = 0
        board = []
        for line in fin:
            if idx==0:
                n = int(line.strip())
            else:
                board.extend([int(i) for i in line.strip().split()])
            idx += 1
    return np.array(board).reshape((n, n))

def main():


    init_board = np.array([[0, 1, 3], [4, 2, 5], [7, 8, 6]])

    path = 'data/puzzle42.txt'
    init_board = read_init(path)
    goal_board = gen_goal_board(init_board.shape[0])

    pf = manhattan_priority
    #pf = hamming_priority

    # init_board = np.array([[8,1,3], [4,0,2], [7,6,5]])

    # print hamming_priority(init_state, goal_board)

    # print manhattan_priority(init_state, goal_board)

    # print neighbour_boards(goal_board)

    start = time.time()
    state, n_states_enq = search(init_board, goal_board, pf=pf, timeout=5)
    end = time.time()

    print
    print path
    print
    print 'Number of states enqueued: {}'.format(n_states_enq)
    print 'Number of moves: {}'.format(state[1])
    print state[0]

    print 'Runtime: {} seconds'.format(end-start)
    print 'Priority funciton: {}'.format(pf)

main()
