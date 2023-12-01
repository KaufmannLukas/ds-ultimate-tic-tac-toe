import logging

import numpy as np

import json

logger = logging.getLogger(__name__)


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
        """
        Represents a player in the Ultimate Tic-Tac-Toe game.

        This class holds the state and history of a player's moves, including the
        current state of their board, the wins in each of the 9 sub-boards, and
        the player's color.

        Attributes:
            history (list): A list to record the history of the player's moves.
            board (numpy.ndarray): A 9x9 numpy array representing the game board,
                where each element indicates whether the cell is occupied by the player.
            wins (numpy.ndarray): A 1D numpy array of length 9, where each element
                represents whether the player has won in the corresponding sub-board.
            color: The color assigned to the player, used to differentiate between players.

        Args:
            color: An identifier for the player's color or symbol.
        """

        def __init__(self, color):
            """
            Initializes a new player with a given color.

            The player's history is initialized as an empty list, the board is initialized
            as a 9x9 matrix of zeros (indicating no moves made), and the wins array is
            initialized indicating no sub-board wins.

            Args:
                color: An identifier for the player's color or symbol.
            """
            self.history = []
            self.board = np.zeros((9, 9), dtype=bool)
            self.wins = np.zeros((9,), dtype=bool)
            self.color = color

        def copy(self):
            """
            Creates a copy of this player instance.

            Returns a new _Player instance with the same state (history, board, wins, color)
            as the current instance, but as a separate object.

            Returns:
                _Player: A new instance of _Player with the same state.
            """
            new_player = Game._Player(self.color)
            new_player.history = self.history.copy()
            new_player.board = np.copy(self.board)
            new_player.wins = np.copy(self.wins)
            return new_player

    def __init__(self):
        """
        Initializes a new game of Ultimate Tic-Tac-Toe.

        This method sets up the game by initializing two players, white and black,
        each represented by an instance of the _Player class. The game starts with
        no winner, not yet finished, and with no moves made.

        Attributes:
            white (Game._Player): The player playing with white pieces, symbolized as "X".
            black (Game._Player): The player playing with black pieces, symbolized as "O".
            winner: Stores the winning player once the game concludes. None if the game is ongoing or a draw.
            done (bool): Indicates whether the game has finished. False initially.
            last_move: Stores the last move made in the game. None initially, and updated as the game progresses.
        """
        self.white = Game._Player(color="white (X)")
        self.black = Game._Player(color="black (O)")
        self.winner = None
        self.done = False
        self.last_move = None
        self.global_draw = False

    # TODO: maybe implement immutable game-states later
    def __key(self):
        return (
            hash(np.ndarray.tobytes(self.white.board)),
            hash(np.ndarray.tobytes(self.black.board)),
            self.winner,
            self.done,
            self.last_move,
        )

    def __hash__(self) -> int:
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()

    @property
    def current_player(self):
        """
        Determines the current player based on the game's history.

        This property checks the length of the move history for each player. If both players 
        have made the same number of moves, it is the white player's turn. Otherwise, 
        it's the black player's turn. This approach assumes the white player always makes 
        the first move.

        Returns:
            _Player: The player (either white or black) whose turn it is to make a move.
        """
        if len(self.white.history) == len(self.black.history):
            return self.white
        return self.black


    @property
    def blocked_fields(self):
        """
        Computes the blocked fields of the Ultimate Tic-Tac-Toe board.

        A field is considered blocked if it's either already occupied by a player or part of
        a sub-game (3x3 grid) that has been won. This property also takes into account the rules
        for sending a player to a specific sub-game based on the last move. If the targeted
        sub-game is completed, the player is free to move in any non-completed sub-game.

        Returns:
            numpy.ndarray: A 9x9 Boolean array where True represents a blocked field and
            False represents a free field. If the last move is None (i.e., at the start of the game),
            all fields are unblocked, indicated by an array of False values.
        """
        if self.last_move is None:
            return np.zeros((9, 9), dtype=bool)

        blocked_fields = self.white.board | self.black.board
        finished_games = (self.white.wins | self.black.wins)[
            :, np.newaxis] * np.ones((9, 9), dtype=bool)
        blocked_fields = blocked_fields | finished_games
    

        # if a whole game is blocked, return the rest of the games as free
        if all(blocked_fields[self.last_move[1]]):
            return blocked_fields

        blocked_games = np.ones((9, 9), dtype=bool)
        blocked_games[self.last_move[1]] = np.zeros((9,), dtype=bool)

        blocked_fields = blocked_fields | blocked_games
        return blocked_fields

    @property
    def complete_history(self):
        """
        Constructs the complete move history of the game in the order the moves were made.

        This property interleaves the moves made by the white and black players to recreate 
        the sequence of moves as they happened in the game. It assumes that the white player 
        makes the first move. The history is represented as a list of moves, where each move 
        is a tuple indicating the position played.

        If the number of moves made by each player is not equal (i.e., white has made one more move), 
        the last move in the list will be from the white player.

        Returns:
            list: A list of up to 81 moves (maximum in a 9x9 Ultimate Tic-Tac-Toe game), 
            where each element is a move tuple or None if the move slot is unoccupied.
        """
        white_moves = self.white.history
        black_moves = self.black.history
        # Interleave the moves, assuming white starts first
        moves = [None]*81  # Initialize a list with 81 moves
        moves[:len(white_moves)*2:2] = white_moves
        moves[1:len(black_moves)*2:2] = black_moves
        return moves

    def copy(self):
        new_game = Game()
        # Assuming _Player also has a custom copy method
        new_game.white = self.white.copy()
        new_game.black = self.black.copy()
        new_game.winner = self.winner
        new_game.done = self.done
        new_game.last_move = self.last_move
        return new_game

    def get_valid_moves(self) -> set:
        """
        Identifies and returns all possible valid moves for the current game state.

        This method computes valid moves based on the current state of the board, considering
        which fields are blocked. A field is valid for a move if it is not blocked. The method
        checks the `blocked_fields` property to determine the status of each field on the board.

        Returns:
            set: A set of tuples, where each tuple represents a valid move. Each tuple consists
            of two integers: `game_idx` and `field_idx`. `game_idx` represents the index of the 
            sub-game (3x3 grid) on the board, and `field_idx` represents the specific field index 
            within that sub-game.
        """
        valid_fields = ~self.blocked_fields
        # Get all indexes where the matrix is True
        valid_indexes = np.argwhere(valid_fields)
        return set(map(tuple, valid_indexes))

    def check_valid_move(self, game_idx, field_idx):
        '''
        Returns True if the move (game_idx, field_idx) is valid / not blocked
        '''
        return not self.blocked_fields[game_idx, field_idx]

    def check_win(board):
        """
        Determines whether a winning combination has been achieved on a given 3x3 board.

        This function assesses the provided board against a set of predefined winning combinations.
        A winning combination is achieved if any of these combinations is fully matched on the board.
        The function uses a bitwise AND operation to identify positions on the board that match 
        any winning combination, then checks if any of these combinations are completely filled.

        Args:
            board (numpy.ndarray): A 3x3 numpy array representing a Tic-Tac-Toe board, where each cell
            is a boolean indicating whether it is occupied by the current player.

        Returns:
            bool: True if a winning combination is found, False otherwise.
        """
        win_combs = Game.WINNING_COMBINATIONS  # get all combinations to win a game
        win_mask = win_combs & board  # get just the moves that hits a winning combination
        # check if all hits for a winning combination were made
        win_list = np.all(win_mask == win_combs, axis=1)
        return np.any(win_list)  # determine if a winning combination was hit

    def play(self, game_idx, field_idx):
        """
        Executes a move in the game, updates the game state, and checks for wins or draw.

        This method allows a player to make a move at the specified indices, updates the player's
        history and board state, and then checks for any local or global wins. If a move is made on a 
        blocked field, a ValueError is raised. The method also checks for a draw if all fields are blocked.

        Args:
            game_idx (int): The index of the sub-game (3x3 grid) where the move is made.
            field_idx (int): The index of the field within the specified sub-game where the move is made.

        Returns:
            tuple: A tuple of two boolean values:
                - The first boolean indicates if there is a local game win (True if won, False otherwise).
                - The second boolean indicates if there is a global game win (True if won, False otherwise).

        Raises:
            ValueError: If a move is attempted on a blocked field.
        """
        if not self.check_valid_move(game_idx, field_idx):
            print(self)
            raise ValueError(
                f"You tried to play on a blocked field ({game_idx}, {field_idx})!")

        current_player = self.current_player
        current_player.history.append((game_idx, field_idx))
        self.last_move = (game_idx, field_idx)
        current_player.board[game_idx, field_idx] = True

        win_local_game = False
        win_global_game = False

        if Game.check_win(current_player.board[game_idx]):
            current_player.wins[game_idx] = True
            win_local_game = True
            # print(f"{current_player.color} wins local game {game_idx}")
            if Game.check_win(current_player.wins):
                win_global_game = True
                self.winner = current_player.color
                # print(f"{current_player.color} wins global game")
                self.done = True

        # check for a draw
        if np.all(self.blocked_fields):
            if not win_global_game:
                self.global_draw = True
                #print("drawwwwwwwwwwwww!!!!!!!!!")
            self.done = True

        return win_local_game, win_global_game


    def get_index_from_vector(index: int):
        field_idx = index % 9 # -> field index
        game_idx = index // 9

        return game_idx, field_idx
    

    def check_draw(self):
        global_draw = self.global_draw

        local_wins = self.white.wins | self.black.wins
        played_fields = self.white.board | self.black.board
        full_games = played_fields.sum(axis=1) == 9

        local_draws = (full_games ^ local_wins) > 0

        return global_draw, local_draws
        



    def _reshape_board(board):
        """
        Reshapes a 9x9 Ultimate Tic-Tac-Toe board for visual representation purposes.

        The original 9x9 board is structured such that each row represents a single game (3x3 sub-board),
        and each column within that row represents a field within that game. This method rearranges the board
        so that it can be visually represented in a format where each 3x3 block corresponds to a sub-game.
        In the reshaped board, the first row of a whole game (3x3 block) is composed of the first rows of the
        first three games, and so on, effectively grouping the sub-games into their visual representation.

        Args:
            board (numpy.ndarray): A 9x9 numpy array representing the original state of the board, where each
            row corresponds to a single 3x3 sub-game.

        Returns:
            numpy.ndarray: A 9x9 numpy array representing the reshaped board suitable for visual representation,
            with each 3x3 block corresponding to one of the nine sub-games.
        """
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

        blank_board = np.zeros((9,9))
        blank_board[self.last_move] = True
        last_move = Game._reshape_board(blank_board)


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
                if last_move[i, j]:
                    row[-1] = '\033[1m' + row[-1] + '\033[0m'
                    #row.append('\033[0m')
            if i in (3, 6):
                rows.append("\n")
            rows.append("  ".join(row) + "\n")

        repr = "".join(rows)

        # reorder axis
        return repr
    

    # def make_json(self):
    #     json_data = {
    #         # TODO: add draw (None can be draw or just an unfinished game currently)
    #         "win_global_game": "white" if self.winner == "white" else ("black" if self.winner == "black" else "None"),
    #         "black_history": self.black.history,
    #         "white_history": self.white.history,
    #         "combined_history": self.complete_history,
    #         "last_move": self.last_move,
    #         "game_state": {
    #             "games": {}
    #         }
    #     }


    #     for game_idx in range(9):
    #         json_data["game_state"]["games"][f"game_{game_idx}"] = {
    #             "won_by": "white" if self.white.wins[game_idx] else (
    #                 "black" if self.black.wins[game_idx] else "None" # TODO: add draw
    #             ),
    #             "next_move": game_idx == self.last_move[1] if self.last_move else False,
    #             "fields": {}
    #         }

    #         for field_idx in range(9):
    #             field_key = f"field_{field_idx}"
    #             json_data["game_state"]["games"][f"game_{game_idx}"]["fields"][field_key] = {
    #                 "white": bool(self.white.board[game_idx, field_idx]),
    #                 "black": bool(self.black.board[game_idx, field_idx]),
    #                 "last_move": bool((game_idx, field_idx) == self.last_move) if self.last_move else False,
    #                 "blocked_field": bool(self.blocked_fields[game_idx, field_idx]),
    #                 "valid_move": bool(not self.blocked_fields[game_idx, field_idx])
    #             }

    #     json_string = json.dumps(json_data, indent=4, default=lambda x: bool(x))
    #     # print(json_string)
    #     return json_string


    def make_json(self):
        global_draw, local_draws = self.check_draw()
        local_wins_white = int(sum(self.white.wins))
        local_wins_black = int(sum(self.black.wins))

        json_data = {
            "global_win": "draw" if self.global_draw else (
                "white" if self.winner == self.white.color else (
                    "black" if self.winner == self.black.color else "None"
                )),
            "current_player": self.current_player.color,
            "local_wins_white": local_wins_white, #if local_wins_white > 0 else 0,
            "local_wins_black": local_wins_black, #if local_wins_black > 0 else 0,
            "games": {}
        }

        for game_idx in range(9):
            json_data["games"][f"game_{game_idx}"] = {
                "won_by": "white" if self.white.wins[game_idx] else (
                        "black" if self.black.wins[game_idx] else "None"
                    ),
                "next_move": game_idx in {move[0] for move in self.get_valid_moves()},
                "fields": {}
            }

            for field_idx in range(9):
                field_key = f"field_{field_idx}"
                json_data["games"][f"game_{game_idx}"]["fields"][field_key] = {
                    "white": bool(self.white.board[game_idx, field_idx]),
                    "black": bool(self.black.board[game_idx, field_idx]),
                    "last_move": bool((game_idx, field_idx) == self.last_move) if self.last_move else False,
                    "blocked_field": bool(self.blocked_fields[game_idx, field_idx]),
                    "valid_move": bool(not self.blocked_fields[game_idx, field_idx])
                }

        #json_string = json.dumps(json_data, indent=4, default=lambda x: bool(x))
        # print(json_string)
        #return json_string
        return json_data

# Example usage
# game = Game()
# json_string = game.make_json()
# with open("json_test.json", "w") as file:
#     file.write(json_string)

# game = Game()
# json_test = game.make_json()
# json_print = json.dumps(json_test, indent=4, default=lambda x: bool(x))
# print(json_print)
