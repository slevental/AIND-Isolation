"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import itertools
from unionfind import UnionFind

MAX_DEPTH = 16
TIME_THRESHOLD = 100


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return connected_components_with_backoff(game, player)


def connected_components_with_backoff(game, player):
    """
    Combination of two heuristics with back-off from one to another

    :param game:
    :param player:
    :return:
    """
    score = connected_components_score(game, player)
    return weighted_number_of_moves(game, player) if score == 0 else score


def weighted_number_of_moves(game, player):
    return float(len(game.get_legal_moves(player))) \
           - float(len(game.get_legal_moves(game.get_opponent(player)))) * 1.5


def connected_components_score(game, player):
    return float(connected_components_diff(game, player))


def baseline(game, player):
    return float(len(game.get_legal_moves(player))) \
           - float(len(game.get_legal_moves(game.get_opponent(player))))


def neighbors(game, position):
    """
    Getting neighbors positions from given one

    :param game:
    :param position:
    :return:
    """
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    return [(position[0] + x, position[1] + y)
            for (x, y) in directions
            if game.move_is_legal((position[0] + x, position[1] + y))]


def connected_components_diff(game, player):
    """
    Difference between number of connected components
    of one player and its opponent

    :param game:
    :param player:
    :return:
    """
    size = game.width * game.height
    uf = UnionFind(size)
    blank = game.get_blank_spaces()
    for bs in blank:
        for n in neighbors(game, bs):
            uf.union(bs, n)
    player_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    for n in neighbors(game, player_location):
        uf.union(n, player_location)
    for n in neighbors(game, opp_location):
        uf.union(n, opp_location)

    pl_score = float(uf.components(player_location))
    op_score = float(uf.components(opp_location))
    return pl_score - op_score


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        # Reducing time left by some threshold - to prevent failure
        self.time_left = lambda: time_left() - TIME_THRESHOLD

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # Returning in case of no legal moves
        if len(legal_moves) == 0:
            return -1, -1

        # Initializing best move/score
        best_move = legal_moves[0]
        best_score = float("-inf")

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            method = self.minimax if self.method == 'minimax' else self.alphabeta
            if self.iterative:
                for d in range(1, MAX_DEPTH):
                    score, candidate = method(game, d, maximizing_player=True)
                    if score > best_score:
                        best_move = candidate
                        best_score = score
            else:
                score, candidate = method(game, MAX_DEPTH, maximizing_player=True)
                if score > best_score:
                    best_move = candidate


        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        player = game.active_player if maximizing_player else game.inactive_player

        # Retrieving list of legal moves from environment
        legal_moves = game.get_legal_moves(game.active_player)

        # In case of no legal moves - returning (-1, -1)
        if len(legal_moves) == 0:
            return float('-inf'), (-1, -1)

        score = float('-inf') if maximizing_player else float('inf')
        optimal_move = None
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            new_game = game.forecast_move(move)

            # In case of depth more than one - do recursive call
            if depth > 1:
                new_score, _ = self.minimax(new_game, depth - 1, not maximizing_player)
            else:
                new_score = self.score(new_game, player)

            if (maximizing_player and new_score > score) or (not maximizing_player and new_score < score):
                score = new_score
                optimal_move = move

        return score, optimal_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        # Get maximizing player - one we do optimization for
        player = game.active_player if maximizing_player else game.inactive_player

        # Retrieving list of legal moves from environment
        legal_moves = game.get_legal_moves(game.active_player)

        # In case of no legal moves - returning (-1, -1)
        if len(legal_moves) == 0:
            return float('-inf'), (-1, -1)

        # Initializing score and move
        score = float('-inf') if maximizing_player else float('inf')
        optimal_move = None
        random.shuffle(legal_moves)
        for move in legal_moves:
            # Check time and return in case of timeout
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            # Change environment
            new_game = game.forecast_move(move)

            # In case of depth more than one - do recursive call
            if depth > 1:
                new_score, _ = self.alphabeta(new_game, depth - 1, alpha, beta, not maximizing_player)
            else:
                new_score = self.score(new_game, player)

            # If it's a maximizing player - update alpha and choose maximum score
            if maximizing_player:
                alpha = max(alpha, new_score)
                score = max(score, new_score)
                if beta <= alpha:
                    break
            else:
                beta = min(beta, new_score)
                score = min(score, new_score)
                if beta <= alpha:
                    break

            if score == new_score:
                optimal_move = move

        return score, optimal_move
