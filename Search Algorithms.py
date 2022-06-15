#!/usr/bin/env python
# coding: utf-8

# # <center> AI - CA1<br>Name: Shahryar Namdari<br>ID: 810098043<br>9/3/2022<center>

# In[1]:


import pandas as pd
import time
import numpy as np
import queue
import copy
from heapq import heapify, heappush, heappop


# ## Function to read file

# In[2]:


def read_file(file_address,mode):
    my_file = open(file_address,mode)
    line = my_file.readline()
    global field_row
    global field_column
    field_row, field_column = (int(val) for val in line.split())
    #creating the field
    global field
    field = np.zeros((field_row,field_column))

    ### Initial state is Gandalfs position
    #placing gandalf
    line = my_file.readline()
    global gandalfs_row
    global gandalfs_column
    gandalfs_row, gandalfs_column = (int(val) for val in line.split())
    field[gandalfs_row][gandalfs_column] = 7

    ### Goal state is Gondors position
    #placing Gondor
    line = my_file.readline()
    global gondors_row
    global gondors_column
    gondors_row, gondors_column = (int(val) for val in line.split())
    field[gondors_row][gondors_column] = 10

    ### Action and Transition model for Gandalf is to go up, down, left and right. Also to pick up and drop RF.
    #finding num of orcs(k) and RFs(l)
    line = my_file.readline()
    k, l = (int(val) for val in line.split())

    #placing orcs and their areas
    global orc_area_dict
    global x
    global y
    global c
    orc_area_dict = {}
    for t in range(k):
        line = my_file.readline()
        x, y, c = (int(val) for val in line.split())
        field[x][y] = -1
        for i in range(-c,c+1):
            for j in range(-c,c+1):
                if abs(i)+abs(j) <= c :
                    if x+i>=0 and x+i<field_row and y+j>=0 and y+j<field_column and field[x+i][y+j] != -1:
                        field[x+i][y+j] = -2-t
                        orc_area_dict[(x+i,y+j)] = c

    #placing RFs
    global RFs_list
    RFs_list = []
    for _ in range(l):
        line = my_file.readline()
        row, column = (int(val) for val in line.split())
        new_RF = (row,column)
        RFs_list.append(new_RF)
        field[row][column] = 8

    #placing castles
    global RF_dict
    global castles_list
    RF_dict = {}
    castles_list = []
    for t in range(l):
        line = my_file.readline()
        row, column = (int(val) for val in line.split())
        new_castle = (row,column)
        castles_list.append(new_castle)
        RF_dict[RFs_list[t]] = new_castle
        field[row][column] = 9

    my_file.close()


# ### Classes to help algorithm

# In[3]:


class Problem(object):
    def __init__(self):
        self.goal = (gondors_row, gondors_column)
        self.gandalf = (gandalfs_row, gandalfs_column)
        self.start_state = (gandalfs_row, gandalfs_column)
        #self.have_RF = False
        
    def goal_test(self, node):
        if (self.goal == node.get_state()) and (len(node.get_dropped_RF()) == len(RFs_list)):
            return True
        else:
            return False

    def get_start_state(self):
        return self.start_state
    
    def can_go_first(self, state, orc_area_count):
        if state[0] < 0 or state[0] >= field_row or         state[1] < 0 or state[1] >= field_column:
            return False
        #print(state,field_row,field_column)
        if field[state] == -1:
            return False
        return True

    def can_go_second(self, state, orc_area_count):
        if field[state] <= -2 and orc_area_dict[state] < (orc_area_count):
            return False
        return True
             
        
class Node(object):
    def __init__(self, position):
        self.state = tuple(position)
        self.parent = None
        self.orc_area_count = 0
        self.depth = 0
        self.path = []
        self.have_RF = tuple()
        self.dropped_RF = set()
        self.fn = 0
        
    def __hash__(self):
        return hash((self.state, self.have_RF, str(self.dropped_RF)))
    def __lt__(self, other):
        return self.fn < other.fn
    def __gt__(self, other):
        return self.fn > other.fn
    def __eq__(self, other):
        return self.fn == other.fn
    
    def get_state(self):
        return self.state

    def DLS_hash(self):
        return hash((self.depth, self.state, self.have_RF, str(self.dropped_RF)))
    
    def get_depth(self):
        return self.depth
    def set_depth(self, number):
        self.depth = number
        
    def set_parent(self, node_):
        self.parent = node_
    def get_parent(self):
        return self.parent

    def get_have_RF(self):
        return self.have_RF
    def remove_have_RF(self):
        self.have_RF = tuple()
    def set_have_RF(self, position):
        self.have_RF = position

    def get_dropped_RF(self):
        return self.dropped_RF
    def set_dropped_RF(self, set_):
        self.dropped_RF = set_
    def add_dropped_RF(self, position):
        (self.dropped_RF).add(position)

    def set_path(self, path_):
        self.path = path_
    def get_path(self):
        return self.path

    def get_orc_area_count(self):
        return self.orc_area_count
    def set_orc_area_count(self, number):
        self.orc_area_count = number

    def get_fn(self):
        return self.fn
    def set_fn(self, number):
        self.fn = number


# ## BFS Function

# Breadth First Search is a traversal technique in which we traverse all the nodes of the graph in a breadth-wise motion. In BFS, we traverse one level at a time and then jump to the next level. In a graph, the traversal can start from any node and cover all the nodes level-wise.<br>BFS is complete and it will give us optimal solution with time complexity of O(b^d)

# In[4]:


def BFS():
    problem = Problem()
    seen_state_count = 1
    start_node = Node(problem.get_start_state())
    if problem.goal_test(start_node):
        return 1, start_node.get_path(), seen_state_count
    
    #in next line we have movements(down, right, up and left). also [1,1] means pick RF & [0,0] means drop RF.
    pick_ = [1,1]
    drop_ = [0,0]
    D = [1,0] #down
    R = [0,1] #right
    U = [-1,0] #up
    L = [0,-1] #left
    possible_actions = [pick_,drop_,D,L,U,R]
    
    frontier_queue = queue.Queue()
    frontier_set = set()
    frontier_queue.put(start_node)
    frontier_set.add(hash(start_node))
    
    while(frontier_queue):
        node = frontier_queue.get()
        for action in possible_actions:
            #pick
            if action == pick_ and (node.get_state() in RFs_list) and (not node.get_have_RF())             and (RF_dict[node.get_state()] not in node.get_dropped_RF()):
                child = Node([copy.copy(node.get_state()[0]), copy.copy(node.get_state()[1])])
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
                child.set_path(copy.copy(node.get_path()))
                child.set_have_RF(copy.copy(child.get_state()))
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                seen_state_count += 1
                frontier_queue.put(child)
                frontier_set.add(hash(child))
                continue

            #drop
            if action == drop_ and (node.get_state() in castles_list) and (node.get_have_RF())            and RF_dict[node.get_have_RF()] == node.get_state():
                child = Node([copy.copy(node.get_state()[0]), copy.deepcopy(node.get_state()[1])])
                #child.set_parent(node)
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
                child.set_path(copy.copy(node.get_path()))
                child.set_have_RF(tuple())
                #problem.have_RF = False
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                child.add_dropped_RF(copy.copy(node.get_state()))
                seen_state_count += 1
                frontier_queue.put(child)
                frontier_set.add(hash(child))
                continue
            if action == [0,0] or action == [1,1]:
                continue
            
            ##movement
            child = Node([copy.copy(node.get_state()[0]) + copy.copy(action[0])                          , copy.copy(node.get_state()[1]) + copy.copy(action[1])])
            child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
            
            #Handle orc area
            if not problem.can_go_first(child.get_state(), child.get_orc_area_count()):
                continue
            if (field[child.get_state()] <= -2) and (field[child.get_state()] == field[node.get_state()]):
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()) + 1)
            elif (field[child.get_state()] <= -2) and (field[node.get_state()] <= -2)                     and (field[child.get_state()] != field[node.get_state()]):
                child.set_orc_area_count(1)
            elif (field[child.get_state()] <= -2) and (field[node.get_state()] > -2):
                child.set_orc_area_count(1)
            else:
                child.set_orc_area_count(0)
                    
            if problem.can_go_second(child.get_state(), child.get_orc_area_count()):
                child.set_have_RF(copy.copy(node.get_have_RF()))
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                
                #print(node.get_path())
                if action == D:
                    path = copy.copy(node.get_path())
                    path.append('D')
                    child.set_path(path)
                if action == R:
                    path = copy.copy(node.get_path())
                    path.append('R')
                    child.set_path(path)
                if action == L:
                    path = copy.copy(node.get_path())
                    path.append('L')
                    child.set_path(path)
                if action == U:
                    path = copy.copy(node.get_path())
                    path.append('U')
                    child.set_path(path)

                if hash(child) not in frontier_set:
                    seen_state_count += 1
                    if problem.goal_test(child):
                        return 1, child.get_path(), seen_state_count
                    frontier_queue.put(child)
                    frontier_set.add(hash(child))
    return 0, 0, 0


# #### Running BFS

# In[6]:


def run_three_times(func, test, file_address, read_mode):
    read_file(file_address, read_mode)
    start1 = time.time()
    func()
    end1 = time.time()
    t1 = abs(end1 - start1)*1000
    start1 = time.time()
    func()
    end1 = time.time()
    t2 = abs(end1 - start1)*1000
    start1 = time.time()
    a,b,c = func()
    end1 = time.time()
    t3 = abs(end1 - start1)*1000
    t_mean = (t1 + t2 + t3)/3
    print('test ',test,':')
    if a:
        print('Solved!')
        print('path: ', ''.join(b))
        print('path length: ', len(b))
        print('seen state: ', c)
        print('t_mean: ', t_mean, ' ms')
        print(' ')
    else:
        print("can't solve!")
        print(' ')
run_three_times(BFS,'00','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\newTest.txt','r')    
# run_three_times(BFS,'00','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_00.txt','r')
# run_three_times(BFS,'01','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_01.txt','r')
# run_three_times(BFS,'02','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_02.txt','r')
# run_three_times(BFS,'03','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_03.txt','r')


# ## DLS Function

# In[6]:


def DLS(search_depth):
    problem = Problem()
    seen_state_count = 1
    start_node = Node(problem.get_start_state())
    if problem.goal_test(start_node):
        return 1, start_node.get_path(), seen_state_count
    
    #in next line we have movements(down, right, up and left). also [1,1] means pick RF & [0,0] means drop RF.
    pick_ = [1,1]
    drop_ = [0,0]
    D = [1,0] #down
    R = [0,1] #right
    U = [-1,0] #up
    L = [0,-1] #left
    possible_actions = [pick_,drop_,D,L,U,R]
    
    frontier_stack = []
    frontier_set = set()
    frontier_stack.append(start_node)
    frontier_set.add(start_node.DLS_hash())

    while(frontier_stack):
        node = frontier_stack.pop()
        if problem.goal_test(node):
            return 1, node.get_path(), seen_state_count
        if((node.get_depth() + 1) > search_depth):
            continue #**** it should be continue!
        for action in possible_actions:
            #pick
            if action == pick_ and (node.get_state() in RFs_list) and (not node.get_have_RF())             and (RF_dict[node.get_state()] not in node.get_dropped_RF()):
                child = Node([copy.copy(node.get_state()[0]), copy.copy(node.get_state()[1])])
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
                child.set_path(copy.copy(node.get_path()))
                child.set_have_RF(copy.copy(child.get_state()))
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                child.set_depth(node.get_depth() + 1)

                seen_state_count += 1
                frontier_stack.append(child)
                frontier_set.add(child.DLS_hash())
                continue

            #drop
            if action == drop_ and (node.get_state() in castles_list) and (node.get_have_RF())            and RF_dict[node.get_have_RF()] == node.get_state():
                child = Node([copy.copy(node.get_state()[0]), copy.copy(node.get_state()[1])])
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
                child.set_path(copy.copy(node.get_path()))
                child.set_have_RF(tuple())
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                child.add_dropped_RF(copy.copy(node.get_state()))
                child.set_depth(node.get_depth() + 1)

                seen_state_count += 1
                frontier_stack.append(child)
                frontier_set.add(child.DLS_hash())
                continue
            if action == [0,0] or action == [1,1]:
                continue

            #movement
            child = Node([copy.copy(node.get_state()[0]) + copy.copy(action[0])                          , copy.copy(node.get_state()[1]) + copy.copy(action[1])])
            child.set_orc_area_count(copy.copy(node.get_orc_area_count()))

            #Handle orc area
            if not problem.can_go_first(child.get_state(), child.get_orc_area_count()):
                continue
            if (field[child.get_state()] <= -2) and (field[child.get_state()] == field[node.get_state()]):
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()) + 1)
            elif (field[child.get_state()] <= -2) and (field[node.get_state()] <= -2)                     and (field[child.get_state()] != field[node.get_state()]):
                child.set_orc_area_count(1)
            elif (field[child.get_state()] <= -2) and (field[node.get_state()] > -2):
                child.set_orc_area_count(1)
            else:
                child.set_orc_area_count(0)

            if problem.can_go_second(child.get_state(), child.get_orc_area_count()):
                child.set_have_RF(copy.copy(node.get_have_RF()))
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                child.set_depth(node.get_depth() + 1)

                #print(node.get_path())
                if action == D:
                    path = copy.copy(node.get_path())
                    path.append('D')
                    child.set_path(path)
                if action == R:
                    path = copy.copy(node.get_path())
                    path.append('R')
                    child.set_path(path)
                if action == L:
                    path = copy.copy(node.get_path())
                    path.append('L')
                    child.set_path(path)
                if action == U:
                    path = copy.copy(node.get_path())
                    path.append('U')
                    child.set_path(path)

                if child.DLS_hash() not in frontier_set:
                    seen_state_count += 1
                    if problem.goal_test(child):
                        return 1, child.get_path(), seen_state_count
                    frontier_stack.append(child)
                    frontier_set.add(child.DLS_hash())

    return 0, 0, 0


# ## IDS Function

# Iterative Deepening Search (IDS) is an iterative graph searching strategy that takes advantage of the completeness of the Breadth-First Search (BFS) strategy but uses much less memory in each iteration (similar to Depth-First Search). <br> It gives us an optimal solution unlike DFS and its time complexity is O(bm).

# In[7]:


def IDS():
    search_depth = 0
    while(True):
        search_depth += 1
        solved, b, c = DLS(search_depth)
        if solved:
            return solved,b,c


# #### Running IDS

# In[8]:


run_three_times(IDS,'00','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_00.txt','r')
run_three_times(IDS,'01','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_01.txt','r')
run_three_times(IDS,'02','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_02.txt','r')
run_three_times(IDS,'03','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_03.txt','r')


# ## A*

# ### Heuristic Function

# Heuristic Function is defined with these situations: <br> 1. when there is no more RF left and you should go to gondor directly<br>2. when RF left and you don't have any RF so you should find the closest RF<br>3. when you have RF so you should go to its castle directly<br> All of these situations defined in an optimal way in the following heuristic function and it obeys the admissibility rule. I used manhattan distance from mean position of RFs, castles and goal in this function. This function is admissible and consistent.

# In[9]:


def heuristic(node, weight):
    sumx = gondors_row
    sumy = gondors_column
    
    for RF in RFs_list:
        sumx += RF[0]
        sumy += RF[1]
    for castle in castles_list:
        sumx += castle[0]
        sumy += castle[1]
        
    X = sumx/(len(RFs_list)+len(castles_list)+1)
    Y = sumy/(len(RFs_list)+len(castles_list)+1)

    dx = node.get_state()[0] - X
    dy = node.get_state()[0] - Y
    return (abs(dx) + abs(dy)) * weight


# ### A* Function

# A* is an optimal informed search algorithm, or a best-first search, meaning that it is formulated in terms of weighted graphs: starting from a specific starting node of a graph, it aims to find a path to the given goal node having the smallest cost (least distance travelled, shortest time, etc.).<br> We will use g(n) as the cost that we have given so far and h(n) as the huristic that estimates the cost from node n to the goal state. <br>Finally f(n) will be h(n)+g(n)
# <br>A* search is complete and its time complexity is O(b^d).

# In[10]:


def A_STAR(weight):
    problem = Problem()
    seen_state_count = 1
    start_node = Node(problem.get_start_state())
    if problem.goal_test(start_node):
        return 1, start_node.get_path(), seen_state_count
    
    #in next line we have movements(down, right, up and left). also [1,1] means pick RF & [0,0] means drop RF.
    pick_ = [1,1]
    drop_ = [0,0]
    D = [1,0] #down
    R = [0,1] #right
    U = [-1,0] #up
    L = [0,-1] #left
    possible_actions = [pick_,drop_,R,D,U,L]
    
    frontier_queue = []
    frontier_set = set()
    heappush(frontier_queue, start_node)
    frontier_set.add(hash(start_node))
    
    while(frontier_queue):
        node = heappop(frontier_queue)
        for action in possible_actions:
            #pick
            if action == pick_ and (node.get_state() in RFs_list) and (not node.get_have_RF())             and (RF_dict[node.get_state()] not in node.get_dropped_RF()):
                child = Node([copy.copy(node.get_state()[0]), copy.copy(node.get_state()[1])])
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
                child.set_path(copy.copy(node.get_path()))
                child.set_have_RF(copy.copy(child.get_state()))
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                child.set_depth(node.get_depth() + 1)
                child.set_fn(heuristic(child, weight) + 2*child.get_depth())
                seen_state_count += 1
                heappush(frontier_queue, child)
                frontier_set.add(hash(child))
                continue

            #drop
            if action == drop_ and (node.get_state() in castles_list) and (node.get_have_RF())            and RF_dict[node.get_have_RF()] == node.get_state():
                child = Node([copy.copy(node.get_state()[0]), copy.deepcopy(node.get_state()[1])])
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
                child.set_path(copy.copy(node.get_path()))
                child.set_have_RF(tuple())
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                child.add_dropped_RF(copy.copy(node.get_state()))
                child.set_depth(node.get_depth() + 1)
                child.set_fn(heuristic(child, weight) + 2*child.get_depth())
                
                seen_state_count += 1
                heappush(frontier_queue, child)
                frontier_set.add(hash(child))
                continue
            if action == [0,0] or action == [1,1]:
                continue
            
            ##movement
            child = Node([copy.copy(node.get_state()[0]) + copy.copy(action[0])                          , copy.copy(node.get_state()[1]) + copy.copy(action[1])])
            child.set_orc_area_count(copy.copy(node.get_orc_area_count()))
            
            #Handle orc area
            if not problem.can_go_first(child.get_state(), child.get_orc_area_count()):
                continue
            if (field[child.get_state()] <= -2) and (field[child.get_state()] == field[node.get_state()]):
                child.set_orc_area_count(copy.copy(node.get_orc_area_count()) + 1)
            elif (field[child.get_state()] <= -2) and (field[node.get_state()] <= -2)                     and (field[child.get_state()] != field[node.get_state()]):
                child.set_orc_area_count(1)
            elif (field[child.get_state()] <= -2) and (field[node.get_state()] > -2):
                child.set_orc_area_count(1)
            else:
                child.set_orc_area_count(0)
                    
            if problem.can_go_second(child.get_state(), child.get_orc_area_count()):
                child.set_have_RF(copy.copy(node.get_have_RF()))
                child.set_dropped_RF(copy.copy(node.get_dropped_RF()))
                child.set_depth(node.get_depth() + 1)
                child.set_fn(heuristic(child, weight) + 2*child.get_depth())
                
                #print(node.get_path())
                if action == D:
                    path = copy.copy(node.get_path())
                    path.append('D')
                    child.set_path(path)
                if action == R:
                    path = copy.copy(node.get_path())
                    path.append('R')
                    child.set_path(path)
                if action == L:
                    path = copy.copy(node.get_path())
                    path.append('L')
                    child.set_path(path)
                if action == U:
                    path = copy.copy(node.get_path())
                    path.append('U')
                    child.set_path(path)

                if hash(child) not in frontier_set:
                    seen_state_count += 1
                    if problem.goal_test(child):
                        return 1, child.get_path(), seen_state_count
                    heappush(frontier_queue, child)
                    frontier_set.add(hash(child))
                    
    return 0, 0, 0


# #### Running A*

# In[11]:


def run_three_times(func, weight, test, file_address, read_mode):
    read_file(file_address, read_mode)
    start1 = time.time()
    func(weight)
    end1 = time.time()
    t1 = abs(end1 - start1)*1000
    start1 = time.time()
    func(weight)
    end1 = time.time()
    t2 = abs(end1 - start1)*1000
    start1 = time.time()
    a,b,c = func(weight)
    end1 = time.time()
    t3 = abs(end1 - start1)*1000
    t_mean = (t1 + t2 + t3)/3
    print('test ',test,':')
    if a:
        print('Solved!')
        print('path: ', ''.join(b))
        print('path length: ', len(b))
        print('seen state: ', c)
        print('t_mean: ', t_mean, ' ms')
        print(' ')
    else:
        print("can't solve!")
        print(' ')
        
run_three_times(A_STAR, 1,'00','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_00.txt','r')
run_three_times(A_STAR, 1,'01','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_01.txt','r')
run_three_times(A_STAR, 1,'02','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_02.txt','r')
run_three_times(A_STAR, 1,'03','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_03.txt','r')


# ## Weighted A*

# It is exactly like A* but we just multuply h(n) by weight!

# ### weight : 5

# In[12]:


weight = 5
run_three_times(A_STAR, weight,'00','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_00.txt','r')
run_three_times(A_STAR, weight,'01','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_01.txt','r')
run_three_times(A_STAR, weight,'02','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_02.txt','r')
run_three_times(A_STAR, weight,'03','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_03.txt','r')


# ### weight : 15

# In[13]:


weight = 15
run_three_times(A_STAR, weight,'00','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_00.txt','r')
run_three_times(A_STAR, weight,'01','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_01.txt','r')
run_three_times(A_STAR, weight,'02','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_02.txt','r')
run_three_times(A_STAR, weight,'03','D:\\term8\\AI\\Fadayi\\CA1\\sample_testcases\\test_03.txt','r')


# In[14]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://res.cloudinary.com/dfjbabeum/image/upload/v1646996341/AI/photo_2022-03-11_14-28-08_cvkwkp.jpg")


# In[15]:


Image(url= "https://res.cloudinary.com/dfjbabeum/image/upload/v1646996391/AI/photo_2022-03-11_14-29-41_vtzomy.jpg")


# In[16]:


Image(url= "https://res.cloudinary.com/dfjbabeum/image/upload/v1646996343/AI/photo_2022-03-11_14-28-18_hcifkk.jpg")


# In[17]:


Image(url= "https://res.cloudinary.com/dfjbabeum/image/upload/v1646996343/AI/photo_2022-03-11_14-28-21_m8ygly.jpg")

