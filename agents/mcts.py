import numpy as np

from environments.game import Game


def ucb_score(parent, child):
    pass


class Node:
    '''
    Each node should store information such as the number of visits, the total reward, and the possible actions
    '''
    def __init__(self, parent):
        self.num_visits = 0
        self.total_reward = 0
        possible_actions = []
        self.childs = set()
        self.parent = parent


    def select():
        '''
        Develop a selection strategy (e.g., Upper Confidence Bound) to choose nodes in the tree to explore further.
        '''
        pass


    def expand():
        '''
        When a selected node has unexplored actions, expand the tree by creating child nodes for those actions.
        '''
        pass


    def simulate():
        '''
        (Rollout): Simulate random playouts from a node to estimate the value of unexplored states. The rollout policy can be random or based on heuristics.
        '''
        pass


    def backpropagate():
        '''
        Update the statistics of nodes as you backpropagate the results of rollouts to their parent nodes.
        '''
        pass



class MCTS:
    '''
    '''
    pass