import logging
import os
import pickle
from math import log, sqrt
from random import sample
from socket import timeout
from time import time

import numpy as np
from tqdm import tqdm

from agents.agent import Agent
from environments.game import Game

logger = logging.getLogger(__name__)

parameters = {
    "C": 2,  # exploration parameter
}


def ucb_score(child):
    '''
    Calculate the Upper Confidence Bound (UCB) score for a given child node.

    Args:
        child (Node): The child node for which to calculate the UCB score.

    Returns:
        float: The UCB score for the given child node.    
    
    Parameters:
        w_i stands for the number of wins for the node considered after the i-th move
        s_i stands for the number of simulations for the node considered after the i-th move
        s_p stands for the total number of simulations after the i-th move run by the parent node of the one considered
        c is the exploration parameter—theoretically equal to √2; in practice usually chosen empirically
    
    '''
    w = child.value_sum
    s = child.visit_count
    s_p = child.parent.visit_count
    c = parameters['C']
    return w/s + c*sqrt(log(s_p)/s)


class Node:
    '''
    Class representing a node in the Monte Carlo Tree Search (MCTS).
    
    Each node stores information such as the number of visits, the total reward, and the possible actions (see below).
    
    Attributes:
        game (Game): The game state associated with the node.
        visit_count (int): The number of times the node has been visited.
        value_sum (float): The sum of values obtained during visits.
        possible_actions (list): List of possible actions in the current game state.
        children (list): List of child nodes.
        parent (Node): The parent node.
        is_fully_expanded (bool): Flag indicating if the node is fully expanded.

    '''

    def __init__(self, game: Game = None, parent: 'Node' = None):
        '''
        Initialize a Node in the Monte Carlo Tree Search.

        Args:
            game (Game): The game state associated with the node.
            parent (Node): The parent node.

        '''    
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

    # TODO: not used yet, delete later maybe? 
    # => Still not used anywhere, recommendation: delete
    def __hash__(self) -> int:
        '''
        Compute the hash value for the node based on the game state.

        Returns:
            int: The hash value of the game state.

        '''    
        return hash(self.game)

    def select_child(self) -> 'Node':
        '''
        Select a child node for exploration based on the Upper Confidence Bound (UCB) score.

        Returns:
            Node: The selected child node.

        '''
        # checks if there are children already to choose from
        if len(self.children) == 0:
            return None

        # returns child with highest ucb-score
        best_child = max(self.children, key=ucb_score)
        return best_child

    def expand(self):
        '''
        Expand the node by creating child nodes for unexplored actions.

        Returns:
            Node: The newly created child node.

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
        # => not necessary, recommendation: delete
        if len(unexplored_moves) == 1:
            self.is_fully_expanded = True

        return new_child

    def backpropagate(self, winner):
        '''
        Update the statistics of nodes during backpropagation.

        Args:
            winner (str): The winner of the game (color of the winning player).

        Note: The results of the rollout are backpropagated to update the values of the nodes along the path
        from the root to the newly expanded node - moves of the simulation are NOT updated.
        '''
        # visit_count for root node
        self.visit_count += 1
        if self.parent is None:
            # Handle the root node separately
            if winner is not None:
                if self.game.current_player.color == winner:
                    value_update = 1.0  # Win
                else:
                    value_update = 0.0  # Loss
            else:
                value_update = 0.5  # Draw

            self.value_sum += value_update
            return None

        # Update based on the outcome (win, draw, loss)
        if winner is not None:
            if self.parent.game.current_player.color == winner:
                value_update = 1.0  # Win
            else:
                value_update = 0.0  # Loss
        else:
            value_update = 0.5  # Draw

        self.value_sum += value_update
        self.parent.backpropagate(winner)


def count_nodes(node: 'Node'):
    '''counts number of nodes in MCTS Tree'''
    if len(node.children) == 0:
        return 1
    return 1 + sum(map(count_nodes, node.children))


def count_leaves(node: 'Node'):
    '''counts number of leaf nodes in MCTS Tree'''
    if len(node.children) == 0:
        return 1
    return sum(map(count_leaves, node.children))

# TODO: implement new functions
# could be useful, recommendation: keep it
def deepest_path(): # counts number of moves of deepest path
    pass

def winner_distribution(): # value sum / visit count from root node
    pass


def simulate(game: Game):
    '''
    Simulate random playouts from a node to estimate the value of unexplored states.

    Args:
        game (Game): The game state to start the simulation from.

    Returns:
        str: The color of the winning player (None for a draw).

    '''
    # check if game is done (winner or draw)
    if game.done:
        return game.winner

    # TODO: maybe later play x random moves at once
    # could be useful, recommendation: keep it
    random_move = sample(sorted(game.get_valid_moves()), 1)[0]

    # simulate a random game for the new child
    new_game = game.copy()
    new_game.play(*random_move)
    return simulate(new_game)



class MCTS(Agent):
    '''
    Agent class for Monte Carlo Tree Search.
    '''

    def __init__(self, memory_path=None, update_memory=False, default_num_iterations = 1000, max_time=0):
        '''
        Initialize the MCTS agent.

        Args:
            memory_path (str): The path to the memory file for storing the MCTS tree.
            update_memory (bool): If True, the memory file will be overwritten and updated.
            default_num_iterations (int): The default number of iterations for MCTS.
            max_time (float): The maximum time (in seconds) for MCTS calculations.

        '''
        logger.info("MCTS agent initialized")
        super().__init__()

        self.num_iterations = default_num_iterations
        self.max_time = max_time

        # memory: represents the node itself, but also contains their children with their values.
        logger.info("load memory")
        # memory_path represents location of stored memory file.
        self.memory_path = memory_path
        # update_memory: if True, memory file will be overwritten and updated.
        self.update_memory = update_memory
        self.memory = Node()
        # check if memory_path exists.
        if memory_path is not None:
            if os.path.exists(memory_path):
                self.load_memory()

    def find_node_by_game(self, game: Game):
        """
        Find the node in the memory tree corresponding to the given game.

        Args:
            game (Game): The game instance for which to find the corresponding node.

        Returns:
            Node: The node in the memory tree that corresponds to the given game.
            Returns and creates new_child if the node is not found.
        """

        node = self.memory

        # TODO: add self.current node to find the node faster.
        # => not sure, recommendation: n/a
        for move in game.complete_history:
            found_child = None
            for child in node.children:
                if child.game.last_move == move:
                    found_child = child
                    break

            if found_child is not None:
                node = found_child
            else:
                # Create a new node only if the move is not found
                new_child = Node(game=game, parent=node)
                node.children.append(new_child)
                node = new_child

        return node

    def play(
            self,
            game: Game,
            num_iterations=None,
            max_time=None,
            disable_progress_bar=True,
    ) -> tuple:
        '''
        Select the best move using Monte Carlo Tree Search (MCTS).

        Args:
            game (Game): The current game state.
            num_iterations (int): The number of iterations for MCTS.
            max_time (float): The maximum time (in seconds) for MCTS calculations.
            disable_progress_bar (bool): If True, disable the progress bar.

        Returns:
            tuple: The selected move.

        '''
        logger.info("Start play")
        start_time = time()

        # find the current game state in memory
        root = self.find_node_by_game(game)
        current_node = root
        logger.debug(f"current_node: \n{current_node}")

        # play game for x number of iterations
        for _ in tqdm(range(num_iterations or self.num_iterations), disable=disable_progress_bar):
            # restart if game is done
            if current_node.game.done:
                current_node = root
                continue
            # expand if not fully explored / expanded
            if not current_node.is_fully_expanded:
                new_child = current_node.expand()
                value_update = simulate(new_child.game)
                new_child.backpropagate(value_update)
                current_node = root

            # select best move
            current_node = current_node.select_child()

            # time limit for calculations
            if (max_time or self.max_time) and (time() - start_time) > (max_time or self.max_time):
                break

        # TODO: check if better way than setting "C" to 0 before choosing best_child
        # => think we can resolve this, recommendation: delete
        # select the best child with a c value of 0
        old_C = parameters['C']
        parameters['C'] = 0
        best_child = root.select_child()
        next_move = best_child.game.last_move
        parameters['C'] = old_C

        return next_move

    def save_memory(self, memory_path=None):
        """
        Save the current state of the MCTS tree to a pickle file.

        Args:
            path (str): location to save the memory to.
        """
        logger.info(f"save memory to {self.memory_path}")
        if memory_path is not None:
            path = memory_path
        else:
            path = self.memory_path
        if self.memory is not None:
            with open(path, 'wb') as file:
                pickle.dump(self.memory, file)
                logger.info(f"Memory saved to {path}")
        else:
            logger.warn("No memory to save.")

    def load_memory(self):
        """
        Load the MCTS tree state from a pickle file.
        """
        if os.path.exists(self.memory_path):
            with open(self.memory_path, 'rb') as file:
                self.memory = pickle.load(file)
                logger.info(f"Memory loaded from {self.memory_path}")
        else:
            logger.info(f"No memory file found at {self.memory_path}.")

    def train(self, num_iterations, max_time):
        '''
        Train the MCTS agent by running simulations and updating the memory.

        Args:
            num_iterations (int): The number of iterations for MCTS simulations.
            max_time (float): The maximum time (in seconds) for MCTS calculations.

        '''
        game = Game()

        next_move = self.play(game,
                              num_iterations=num_iterations,
                              max_time=max_time,
                              disable_progress_bar=False,
                              )
        
        logger.info(f"next move: {next_move}")

        self.save_memory()

    # TODO: figure this out, since update_memory method is defined here.
    # not needed, recommendation: delete
    # def __del__(self):
    #     logger.info("save memory")
    #     if self.update_memory:
    #         if self.memory_path is not None:
    #             self.save_memory()
    #         else:
    #             logger.warn("no memory path to store the memory!!!")
