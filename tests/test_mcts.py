import unittest
from typing import assert_type

import numpy as np

from environments.game import Game
from agents.mcts import MCTS, Node, simulate


full_game_play = [
    (4, 7),
    (7, 7),
    (7, 3),
    (3, 7),
    (7, 1),
    (1, 0),
    (0, 3),
    (3, 2),
    (2, 0),
    (0, 6),
    (6, 5),
    (5, 0),
    (0, 0),
    (0, 4),
    (4, 6),
    (6, 4),
    (4, 5),
    (5, 7),
    (7, 5),
    (5, 1),
    (1, 8),
    (8, 8),
    (8, 2),
    (2, 8),
    (8, 0),
    (0, 7),
    (7, 8),
    (8, 1),
    (1, 4),
    (4, 8),
    (8, 7),
    (7, 4),
    (4, 2),
    (2, 1),
    (1, 2),
    (2, 7),
    (7, 0),
    (0, 5),
    (5, 4),
    (4, 4),
    (4, 3),
    (3, 4),
    (4, 1),
    (1, 1),
    (1, 7),
    (7, 2),
    (2, 4),
    (4, 0),
    (0, 1),
    (1, 5),
    (5, 8),
    (8, 4),
    (3, 1),
    (1, 6),
    (6, 7),
    (7, 6),
    (6, 6),
    (6, 0),
    (0, 8),
    (8, 3),
    (3, 8),
    (8, 6),
    (6, 3),
    (3, 3),
    (3, 6),
    (6, 1),
    (1, 3),
    (3, 0),
    (0, 2),
    (2, 3),
    (3, 5),
    (5, 6),
    (6, 8),
    (8, 5),
    (5, 2),
    (2, 5),
    (5, 3),
    (5, 5),
    (2, 2),
    (2, 6),
]


class TestMCTS(unittest.TestCase):

    def setUp(self):
        """This method is called before each test."""
        pass

    def tearDown(self):
        """This method is called after each test."""
        pass

    def test_select_child(self):
        """..."""
        root = Node()
        root.visit_count = 6

        child1 = Node(parent=root)
        child2 = Node(parent=root)
        child3 = Node(parent=root)

        root.children = [child1, child2, child3]

        child1.visit_count = 1
        child2.visit_count = 2
        child3.visit_count = 3

        child1.value_sum = 0
        child2.value_sum = 1
        child3.value_sum = 3

        self.assertEqual(root.select_child(), child1)

    def test_expand(self):
        """test expanding nodes"""
        root = Node()
        self.assertEqual(len(root.children), 0)
        root.expand()
        self.assertEqual(len(root.children), 1)
        root.expand()
        self.assertEqual(len(root.children), 2)

        game = Game()
        game.play(4, 4)

        root = Node(game)
        for i in range(8):
            root.expand()
        self.assertEqual(len(root.children), 8)
        self.assertTrue(root.is_fully_expanded)
        self.assertIsNone(root.expand())

    def test_simulate(self):
        """This tests a simulation of a game."""
        game = Game()
        result = simulate(game)
        self.assertIn(result, (game.black.color, game.white.color, None))
        full_game = Game()
        for move in full_game_play[:-2]:
            full_game.play(*move)
        result = simulate(full_game)
        self.assertIsNone(result)

    def test_backpropagate(self):
        """This tests the backpropagation process"""
        root = Node()
        self.assertIsNone(root.backpropagate(None))

        child = Node(parent=root)
        child.backpropagate(None)
        self.assertEqual(child.value_sum, 0)
        self.assertEqual(child.visit_count, 1)

        child.backpropagate(child.parent.game.current_player.color)
        self.assertEqual(child.value_sum, 1)
        self.assertEqual(child.visit_count, 2)


if __name__ == '__main__':
    unittest.main()
