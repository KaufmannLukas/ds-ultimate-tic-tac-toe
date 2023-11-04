import numpy as np

class Game:
    WINNING_COMBINATIONS = np.array([
        # rows
        [True, True, True, False, False, False, False, False, False],
        [False, False, False, True, True, True, False, False, False],
        [False, False, False, False, False, False, True, True, True],
        # columns
        [True, False, False, True, False, False, True, False, False],
        [False, True, False, False, True, False, False, True, False],
        [False, False, True, False, False, True, False, False, True],
        # diagonals
        [False, False, True, False, True, False, True, False, False],
        [True, False, False, False, True, False, False, False, True],
    ])


    class _Player:
        def __init__(self, color):
            self.history = []
            self.board = np.zeros((9,9), dtype=bool)
            self.wins = np.zeros((9,), dtype=bool)
            self.color = color


    def __init__(self):
        self.white = Game._Player("white") # white is O and starts
        self.black = Game._Player("black") # black is X
    

    @property
    def current_player(self):
        if len(self.white.history) == len(self.black.history):
            return self.white
        return self.black
    

    @property
    def last_move(self):
        if not self.white.history and not self.black.history:
            return None
        if len(self.white.history) == len(self.black.history):
            return self.black.history[-1]
        else:
            return self.white.history[-1]

    
    @property
    def blocked_fields(self):
        if self.last_move is None:
            return np.zeros((9, 9), dtype=bool)

        blocked_fields = self.white.board | self.black.board
        finished_games = (self.white.wins | self.black.wins)[:, np.newaxis] * np.ones((9, 9), dtype=bool)
        blocked_fields = blocked_fields | finished_games
        
        # if a whole game is blocked, return the rest of the games as free
        if all(blocked_fields[self.last_move[1]]):
            return blocked_fields
        
        blocked_games = np.ones((9,9), dtype=bool)
        blocked_games[self.last_move[1]] = np.zeros((9,), dtype=bool)

        blocked_fields = blocked_fields | blocked_games
        return blocked_fields



    def check_valid_move(self, game_idx, field_idx):
        return not self.blocked_fields[game_idx, field_idx]


    def check_win(board):
        '''
        Get a 3x3 board and checks if there is a winning combination hit.
        '''
        win_combs = Game.WINNING_COMBINATIONS # get all combinations to win a game
        win_mask = win_combs & board # get just the moves that hits a winning combination
        win_list = np.all(win_mask == win_combs, axis=1) # check if all hits for a winning combination were made
        return np.any(win_list) # determine if a winning combination was hit
    

    def play(self, game_idx, field_idx):
        if not self.check_valid_move(game_idx, field_idx):
            raise ValueError(f"You tried to play on a blocked field ({game_idx}, {field_idx})!")
        
        current_player = self.current_player
        current_player.history.append((game_idx, field_idx))
        current_player.board[game_idx, field_idx] = True

        # if self.black.wins[game_idx] == True or self.white.wins[game_idx] == True:
        #     return None

        win_local_game = False
        win_global_game = False

        if Game.check_win(current_player.board[game_idx]):
            current_player.wins[game_idx] = True
            win_local_game = True
            print(f"{current_player.color} wins local game {game_idx}")
            if Game.check_win(current_player.wins):
                win_global_game = True
                print(f"{current_player.color} wins global game") # TODO: what do we make with wins?

        return win_local_game, win_global_game


    def _reshape_board(board):
        # 3x3x9-Array reshape
        reshaped = board.reshape(3, 3, 3, 3)

        # reorder axis
        return reshaped.transpose(0, 2, 1, 3).reshape(9, 9)


    def __repr__(self):
        game = np.zeros((9, 9))
        game[self.black.board] = -1
        game[self.white.board] = 1
        reshaped_board = Game._reshape_board(game)

        blocked_fields = Game._reshape_board(self.blocked_fields)
        
        rows = []
        for i in range(9):

            row = []
            for j in range(9):
                if j in (3, 6):
                    row.append(" ")
                match int(reshaped_board[i, j]):
                    case 1:
                        row.append("X")
                    case -1:
                        row.append("O")
                    case 0:
                        if blocked_fields[i, j]:
                            row.append("·")
                        else:
                            row.append("•")
            if i in (3, 6):
                rows.append("\n")
            rows.append("  ".join(row) + "\n")

        repr = "".join(rows)

        # reorder axis
        return repr
    
