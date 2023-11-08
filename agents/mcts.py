from math import log, sqrt
from random import choice

import numpy as np

from environments.game import Game

parameters = {
    "C": 2,  # exploration parameter
}


def ucb_score(child):
    '''
    w_i stands for the number of wins for the node considered after the i-th move
    s_i stands for the number of simulations for the node considered after the i-th move
    s_p stands for the total number of simulations after the i-th move run by the parent node of the one considered
    c is the exploration parameter—theoretically equal to √2; in practice usually chosen empirically
    '''
    w = child.value_sum   # #simulations of this node wich resulted in a win
    s = child.visit_count  # total # of simulations
    c = parameters["C"]
    s_p = child.parent.visit_count

    return w/s + c*sqrt(log(s_p)/s)


class Node:
    '''
    Each node stores information such as the number of visits, the total reward, and the possible actions.
    '''

    def __init__(self, game: Game = None, parent: 'Node' = None):
        if game is None:
            self.game = Game()
        else:
            self.game = game

        self.visit_count = 0
        self.value_sum = 0
        self.possible_actions = self.game.get_valid_moves()
        self.children = []
        self.parent = parent
        self.is_fully_expanded = False

    def select_child(self):
        '''
        Develop a selection strategy (e.g., Upper Confidence Bound) to choose nodes in the tree to explore further.
        ---
        MCTS begins by selecting a node to expand from the root. This selection process typically involves a balance
        between exploration and exploitation. It uses heuristics or policies to determine which child node to explore.
        '''
        if len(self.children) == 0:  # checks if the game is won, or if they are no valid moves,
            if self.game.done:      # if this is the case, it makes sense that there are no children.
                return None

        if not self.is_fully_expanded:  # double check later
            self.expand()

        assert len(self.children) > 0   # safety net

        best_child = max(self.children, key=ucb_score)
        return best_child   # what are wo doing with the best child?

    def expand(self):
        '''
        When a selected node has unexplored actions, it expands the tree by creating child nodes for those actions.
        ---
        The selected node is expanded by adding one or more child nodes corresponding to possible actions
        that can be taken from the current state. These child nodes are added to the tree.
        '''
        # creates a set of all possible / valid moves for current game / state.
        valid_moves = self.possible_actions

        # keeps track of last moves, that already have been played / explored in this state.
        explored_moves = {child.game.last_move for child in self.children}

        unexplored_moves = valid_moves.difference(explored_moves)
        if len(unexplored_moves) == 0:
            self.is_fully_expanded = True

        # chooses random tuple (move)
        move = choice(unexplored_moves)
        new_game = self.game.copy()
        new_game.play(*move)
        # creates new child for one valid move
        new_child = Node(game=new_game, parent=self)
        self.children.append(new_child)

        return new_child

        # implement change of perspectives (player switch)

    # def expand(self):
    #         action = np.random.choice(np.where(self.expandable_moves == 1)[0])
    #         self.expandable_moves[action] = 0

    #         child_state = self.state.copy()
    #         child_state = self.game.get_next_state(child_state, action, 1)
    #         child_state = self.game.change_perspective(child_state, player=-1)

    #         child = Node(self.game, self.args, child_state, self, action)
    #         self.children.append(child)
    #         return child

    def backpropagate(self, Node, reward):
        '''
        Update the statistics of nodes as you backpropagate the results of rollouts to their parent nodes.
        ---
        The results of the rollout are backpropagated to update the values of the nodes along the path
        from the root to the newly expanded node. This update is based on the outcomes of the simulated episodes.
        '''
        # while Node is not None:
        #     Node.value_sum += reward
        #     Node.visit_count  += 1
        #     Node = Node.parent

    # def backpropogate(self, node, reward):
    #     while node is not None:
    #         node.numVisits += 1
    #         node.totalReward += reward
    #         node = node.parent


#    def _backpropagate(self, path, reward):
#         "Send the reward back up to the ancestors of the leaf"
#         for node in reversed(path):
#             self.N[node] += 1
#             self.Q[node] += reward
#             reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa


def simulate(game: Game):
    '''
    Rollout:
    Simulate random playouts from a node to estimate the value of unexplored states.
    The rollout policy can be random or based on heuristics.
    ---
    After expansion, MCTS performs a simulation, often referred to as a "rollout."
    During a rollout, random actions or actions determined by a simple policy are taken from
    the newly added node to the end of the episode or until a termination condition is met.
    '''
    # simulate a random game for the new child
    if game.done:
        if game.winner is None:
            return 0
        # check rewards for player (white/black, etc.)
        if game.winner.color == "white (X)":
            return 1
        else:
            return -1

    random_move = choice(game.get_valid_moves())

    new_game = game.copy()
    new_game.play(*random_move)
    return simulate(new_game)


class MCTS():
    '''
    '''
    # player=1 ...
    pass
