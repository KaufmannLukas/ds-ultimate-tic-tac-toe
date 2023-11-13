from math import log, sqrt
from random import sample
import logging
from time import time
import pickle


import numpy as np
from tqdm import tqdm

from environments.game import Game
from agents.agent import Agent


logger = logging.getLogger(__name__)

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
    s_p = child.parent.visit_count
    c = parameters['C']
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
        self.possible_actions = self.game.get_valid_moves()  # TODO: Hä
        self.children = []
        self.parent = parent
        self.is_fully_expanded = False

    def select_child(self) -> 'Node':
        '''
        Develop a selection strategy (e.g., Upper Confidence Bound) to choose nodes in the tree to explore further.
        ---
        MCTS begins by selecting a node to expand from the root. This selection process typically involves a balance
        between exploration and exploitation. It uses heuristics or policies to determine which child node to explore.
        '''
        # checks if there are children already to choose from
        if len(self.children) == 0:
            return None

        # returns child with highest ucb-score
        best_child = max(self.children, key=ucb_score)
        return best_child

    def expand(self):
        '''
        When a selected node has unexplored actions, it expands the tree by creating child nodes for those actions.
        ---
        The selected node is expanded by adding one or more child nodes corresponding to possible actions
        that can be taken from the current state. These child nodes are added to the tree.
        '''
        # check if fully expanded already
        if self.is_fully_expanded == True:
            return None

        # creates a set of all possible / valid moves for current game / state.
        valid_moves = self.possible_actions
        # keeps track of last moves, that already have been played / explored in this state.
        explored_moves = {child.game.last_move for child in self.children}
        # checks which moves have not been play / expanded
        unexplored_moves = valid_moves.difference(explored_moves)

        # chooses one random move (tuple)
        move = sample(sorted(unexplored_moves), 1)[0]
        new_game = self.game.copy()
        new_game.play(*move)
        # creates new child for one (valid) move
        new_child = Node(game=new_game, parent=self)
        self.children.append(new_child)

        # TODO: check if there is a better way (update unexplored moves earlier)
        if len(unexplored_moves) == 1:
            self.is_fully_expanded = True

        return new_child

    def backpropagate(self, winner):
        '''
        Update the statistics of nodes as you backpropagate the results of rollouts to their parent nodes.
        ---
        The results of the rollout are backpropagated to update the values of the nodes along the path
        from the root to the newly expanded node. This update is based on the outcomes of the simulated episodes.
        '''
        # visit_count for root node
        # TODO: check if fixed value of 1, maybe. (calculate)
        self.visit_count += 1
        if self.parent is None:
            return None

        # TODO: double check if correct (+/- signs)
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
    # check if game is done (winner or draw)
    if game.done:
        # check if game has a winner or not (draw = [None])
        return game.winner

    # TODO: maybe later play x random moves at once
    random_move = sample(sorted(game.get_valid_moves()), 1)[0]

    # simulate a random game for the new child
    new_game = game.copy()
    new_game.play(*random_move)
    return simulate(new_game)


class MCTS(Agent):
    '''
    player=1
    '''

    def __init__(self, memory=None):
        # memory (if given) is stored as a pickle file and initiated here 
        logger.info("MCTS agent started")
        super().__init__()
        #self.root = root
        #self.current_node = self.root
        '''
        memory: represents the node itself, but also contains their children with their values at the same time
        '''
        self.memory = memory # "big tree"
        '''
        st_memory: only remembers the calculation of the path you're currently playing on
        '''
        #self.st_memory = None # "small tree"

    # TODO: set "C" to 0 before choosing best_child / actual move (done) | check if better way to do that later
    # TODO: set timer (max. 5 sec / move, etc.)
    def play(
            self, 
            game: Game, 
            num_iterations=100, 
            max_time=None, 
            disable_progress_bar=True,
            root = None, # just for training
            ) -> tuple:
        
        start_time = time()

        if self.memory is not None:
            if self.memory.game == game:
                root = self.memory
            else:
                for child in self.memory.children:
                    if game == child.game:
                        root = child
                        break   
        

        current_node = root
        for _ in tqdm(range(num_iterations), disable=disable_progress_bar):
            if current_node.game.done:
                current_node = root
                continue
            if not current_node.is_fully_expanded:
                new_child = current_node.expand()
                value_update = simulate(new_child.game)
                new_child.backpropagate(value_update)
                current_node = root

            best_child = current_node.select_child()
            assert best_child is not None
            current_node = best_child

            if max_time and time() - start_time > max_time:
                break
        

        old_C = parameters['C']
        parameters['C'] = 0
        best_child = root.select_child()
        next_move = best_child.game.last_move
        parameters['C'] = old_C

        self.memory = best_child
        return next_move
    


    def train(self, num_iterations, max_time):
        game = Game()
        root = Node()

        next_move = self.play(game,
                              num_iterations=num_iterations,
                              max_time=max_time,
                              disable_progress_bar=False,
                              root=root)
        logger.info(f"next move: {next_move}")
        with open(f"data/mcts_ltmm.pkl", 'wb') as file:
            pickle.dump(root, file=file)
        





    
