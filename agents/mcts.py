from math import log, sqrt
from random import sample

import numpy as np

from environments.game import Game
from agents.agent import Agent

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
        if len(self.children) == 0:
            return None

        best_child = max(self.children, key=ucb_score)
        return best_child

    def expand(self):
        '''
        When a selected node has unexplored actions, it expands the tree by creating child nodes for those actions.
        ---
        The selected node is expanded by adding one or more child nodes corresponding to possible actions
        that can be taken from the current state. These child nodes are added to the tree.
        '''
        if self.is_fully_expanded == True:
            return None

        # creates a set of all possible / valid moves for current game / state.
        valid_moves = self.possible_actions
        # keeps track of last moves, that already have been played / explored in this state.
        explored_moves = {child.game.last_move for child in self.children}

        unexplored_moves = valid_moves.difference(explored_moves)

        # chooses random tuple (move)
        move = sample(sorted(unexplored_moves), 1)[0]
        new_game = self.game.copy()
        new_game.play(*move)
        # creates new child for one valid move
        new_child = Node(game=new_game, parent=self)
        self.children.append(new_child)

        if len(unexplored_moves) == 1:
            self.is_fully_expanded = True

        return new_child, move

    def backpropagate(self, winner):
        '''
        Update the statistics of nodes as you backpropagate the results of rollouts to their parent nodes.
        ---
        The results of the rollout are backpropagated to update the values of the nodes along the path
        from the root to the newly expanded node. This update is based on the outcomes of the simulated episodes.
        '''
        if self.parent is None:
            self.visit_count += 1
            return None

        self.visit_count += 1

        if winner is not None:
            if self.parent.game.current_player.color == winner:
                self.value_sum += 1
            else:
                self.value_sum -= 1

        self.parent.backpropagate(winner)



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
            return None
        # check for winning player (white/black)
        return game.winner

    random_move = sample(sorted(game.get_valid_moves()), 1)[0]
    # TODO: maybe later play x random moves at once

    new_game = game.copy()
    new_game.play(*random_move)
    return simulate(new_game)




class MCTS(Agent):
    '''
    '''
    # player=1 ...

    def __init__(self):
        self.root = Node()
        super.__init__()

    def play(self, game: Game) -> tuple:
        pass
    
    def search_best_move():
        pass
        

    def train(self, num_iterations):
        
        self.root.select_child()



    # if not self.is_fully_expanded:
    # new_child = self.expand()
    # # -> simulate random game(s) from the new child
    # winner = simulate(new_child.game)
    # # -> backpropagate the result/winner of the simulation back to the root node
    # new_child.backpropagate(winner)


    # import torch

    # from game import Connect2Game
    # from model import Connect2Model
    # from trainer import Trainer

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # args = {
    #     'batch_size': 64,
    #     'numIters': 500,                                # Total number of training iterations
    #     'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    #     'numEps': 100,                                  # Number of full games (episodes) to run during each iteration
    #     'numItersForTrainExamplesHistory': 20,
    #     'epochs': 2,                                    # Number of epochs of training per iteration
    #     'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
    # }

    # game = Connect2Game()
    # board_size = game.get_board_size()
    # action_size = game.get_action_size()

    # model = Connect2Model(board_size, action_size, device)

    # trainer = Trainer(game, model, args)
    # trainer.learn()