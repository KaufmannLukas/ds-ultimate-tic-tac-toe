import unittest
from typing import assert_type

import numpy as np

from environments.game import Game  # Replace with your actual module name
from agents.mcts import MCTS, Node, simulate

# import environments.game


class TestMCTS(unittest.TestCase):

    def setUp(self):
        """This method is called before each test."""
        #self.game = Game()


    def tearDown(self):
        """This method is called after each test."""
        pass  # Add any cleanup code here, if necessary


    def test_ucb_score(self):
        """This tests the output of one ucb_score."""
        pass
    

    def test_select_child(self):
        """..."""
        pass

    def test_expand(self):
        """..."""
        pass


    def test_simulate(self):
        """This tests a simulation of a game."""
        game = Game()
        result = simulate(game)
        self.assertIn(result, (game.black.color, game.white.color, None))


    def test_backpropagate(self):
        """..."""
        pass



    def test_initialization(self):
        """Test that the game is initialized correctly."""
        # self.assertIsInstance(self.game.white, Game._Player)
        # self.assertIsInstance(self.game.black, Game._Player)
        # self.assertEqual(self.game.white.color, "white (X)")
        # self.assertEqual(self.game.black.color, "black (O)")
        # self.assertEqual(len(self.game.white.history), 0)
        # self.assertEqual(len(self.game.black.history), 0)
        # self.assertTrue(np.all(self.game.white.board == 0))
        # self.assertTrue(np.all(self.game.black.board == 0))

    def test_current_player(self):
        """Test the current player property."""
        # self.assertEqual(self.game.current_player.color, "white (X)")
        # self.game.white.history.append((0, 0))
        # self.assertEqual(self.game.current_player.color, "black (O)")



if __name__ == '__main__':
    unittest.main()
