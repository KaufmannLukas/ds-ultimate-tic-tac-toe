import unittest
import numpy as np
from environments.game import Game  # Replace with your actual module name
#import environments.game


class TestGame(unittest.TestCase):
    
    def setUp(self):
        """This method is called before each test."""
        self.game = Game()
        
    def tearDown(self):
        """This method is called after each test."""
        pass  # Add any cleanup code here, if necessary

    def test_initialization(self):
        """Test that the game is initialized correctly."""
        self.assertIsInstance(self.game.white, Game._Player)
        self.assertIsInstance(self.game.black, Game._Player)
        self.assertEqual(self.game.white.color, "white")
        self.assertEqual(self.game.black.color, "black")
        self.assertEqual(len(self.game.white.history), 0)
        self.assertEqual(len(self.game.black.history), 0)
        self.assertTrue(np.all(self.game.white.board == 0))
        self.assertTrue(np.all(self.game.black.board == 0))

    def test_current_player(self):
        """Test the current player property."""
        self.assertEqual(self.game.current_player.color, "white")
        self.game.white.history.append((0, 0))
        self.assertEqual(self.game.current_player.color, "black")

    def test_last_move(self):
        """Test the last move property."""
        self.assertIsNone(self.game.last_move)
        self.game.white.history.append((0, 0))
        self.assertEqual(self.game.last_move, (0, 0))
        self.game.black.history.append((1, 1))
        self.assertEqual(self.game.last_move, (1, 1))

    def test_play(self):
        """Test the play method."""
        self.game.play(0, 0)
        self.assertTrue(self.game.white.board[0, 0])
        self.assertFalse(self.game.black.board[0, 0])
        self.assertEqual(self.game.white.history, [(0, 0)])
        self.assertEqual(len(self.game.black.history), 0)

    def test_illegal_move(self):
        """Test the reaction on illegal moves."""
        self.game.play(4, 4) #w
        # blocked by former move
        self.assertRaises(ValueError, self.game.play, 4, 4)
        # blocked by game restriction
        self.assertRaises(ValueError, self.game.play, 6, 4)

        self.game.play(4, 6) #b
        self.game.play(6, 6) #w
        self.game.play(6, 4) #b
        self.game.play(4, 0) #w
        self.game.play(0, 4) #b
        self.game.play(4, 8) # white winns game 4
        #blocked by finished local game
        self.assertRaises(ValueError, self.game.play, 4, 7)
        
    

if __name__ == '__main__':
    unittest.main()
