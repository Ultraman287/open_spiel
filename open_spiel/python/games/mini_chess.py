# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.

This file contains an implementation of a rudimentary version of mini-chess.
Mini-chess is a simplified version of chess played on a 4x4 board. The game
follows the rules of chess, but with a limited set of pieces and moves.

The implementation includes the game logic, state representation, legal move
generation, and observation functions.

For more information on mini-chess, refer to the official rules and documentation.
"""

import numpy as np

from mini_chess_helper import action_space, space_action, debug_main, debug_alpha_zero, debug_mcts_evaluator
import pyspiel
import numpy as np
import subprocess

'''
Set up the game

Since it's variant of chess, the game is zero-sum, perfect information, and sequential.
'''
_NUM_PLAYERS = 2
_NUM_ROWS = 4
_NUM_COLS = 4
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_GAME_TYPE = pyspiel.GameType(
    short_name="python_mini_chess",
    long_name="Python Mini Chess",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
'''
Num distinct actions is calculated as the 7 possible moves for each of the 4 pawns (28),
plus the 8 possibles moves for each of the 2 bishops (16). The other pieces can all moves
to all 16 squares and so multiplying by 2 for either color gives 32 for each of the rooks,
knights, and kings. The total is 28+16+32+32+32 = 140.

The max chance outcomes is 0 because there is no chance in this game.

The min and max utility are -1 and 1 because the game is zero-sum.

The max game length is used as an estimate to make sure that games that go on too long 
get cut off. The value itself is inspired by the work of Kirill Kryukov on smaller chess variants
and potential moves which can be found at https://kirill-kryukov.com/chess/4x4-chess/nulp.html.
'''
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=28+16+32+32+32,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=50)


class MiniChessGame(pyspiel.Game):
  """A Python version of Mini Chess.
  The class inherets from pyspiel.Game which contains the scaffolding for the basic functions
  that each game implemented within the open spiel framework must have.
  """

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return MiniChessState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state.
      An observer is used to get information about the game state. This is done in the
      form of a tensor which is a 1-D array of floats and a dictionary of views onto the
      tensor. The tensor is indexed by (cell state, row, column). The cell state is a
      one-hot encoding of the piece type and color. The row and column are the position
      of the piece on the board. The dictionary is indexed by strings and is used to
      access the tensor. The strings are the names of the views. The only view is
      "observation" which is the tensor itself.
    """
    
    return BoardObserver(params)



class MiniChessState(pyspiel.State):
  """A state object that implements logic underneath for the working of the game
  It keeps track of the current player, the current board state, and whether or not
  the game is over. It also has functions for getting the current player, the legal
  actions, applying an action, and checking if the game is over.
  ."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.array(
              [['bB', 'bR', 'bK', 'bN'],
              ['.', 'bP1', 'bP2', '.'],
              ['.', 'wP1', 'wP2', '.'],
              ['wB', 'wR', 'wK', 'wN']])
    

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every perfect-information sequential-move game.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player


  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    legal_moves = []

    for i in range(4):
      for j in range(4):
        piece = self.board[i, j]
        if piece == '.':
          continue
        if (piece[0] == 'b' and player == 0) or (piece[0] == 'w' and player == 1):
          continue
        piece_type = piece[1]
        if piece_type == 'P':
          legal_moves += self._legal_pawn_moves(i, j, piece)
        elif piece_type == 'B':
          legal_moves += self._legal_bishop_moves(i, j, piece)
        elif piece_type == 'R':
          legal_moves += self._legal_rook_moves(i, j, piece)
        elif piece_type == 'N':
          legal_moves += self._legal_knight_moves(i, j, piece)
        elif piece_type == 'K':
          legal_moves += self._legal_king_moves(i, j, piece)

    legal_moves.sort()

    return legal_moves

    """
    Each piece has a different set of legal moves and the finding of those moves has been
    abstracted into their own functions. The modularity of this design easily showcases the potential
    for generalization to other chess variants as the important bit is having each indvidual case
    find its own legal moves. The legal moves are then sorted and returned.
    """
  def _legal_pawn_moves(self, i, j, pawn_str):
    legal_moves = []
    opp_str = 'b' if pawn_str[0] == 'w' else 'w'
    for k in range(3):
      row = i - 1 if pawn_str[0] == 'w' else i + 1
      col = j - 1 + k

      if row < 0 or col < 0 or row > 3 or col > 3:
        break
      if self.board[row, col] == '.' or self.board[row, col][0] == opp_str:
        legal_moves.append(action_space[pawn_str+'->('+str(row)+','+str(col)+')'])
    return legal_moves

  def _legal_bishop_moves(self, i, j, bis_str):
    legal_moves = []
    dirs = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    opp_str = 'b' if bis_str[0] == 'w' else 'w'

    for dir in dirs:
      row = i + dir[0]
      col = j + dir[1]
      while row >= 0 and col >= 0 and row < 4 and col < 4:
        if self.board[row, col] == '.':
          legal_moves.append(action_space[bis_str+'->('+str(row)+','+str(col)+')'])
        elif self.board[row, col][0] == opp_str:
          legal_moves.append(action_space[bis_str+'->('+str(row)+','+str(col)+')'])
          break
        else:
          break
        row += dir[0]
        col += dir[1]

    return legal_moves

  def _legal_rook_moves(self, i, j, rook_str):
    legal_moves = []
    dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    opp_str = 'b' if rook_str[0] == 'w' else 'w'

    for dir in dirs:
      row = i + dir[0]
      col = j + dir[1]
      while row >= 0 and col >= 0 and row < 4 and col < 4:
        if self.board[row, col] == '.':
          legal_moves.append(action_space[rook_str+'->('+str(row)+','+str(col)+')'])
        elif self.board[row, col][0] == opp_str:
          legal_moves.append(action_space[rook_str+'->('+str(row)+','+str(col)+')'])
          break
        else:
          break
        row += dir[0]
        col += dir[1]
    return legal_moves

  def _legal_knight_moves(self, i, j, knight_str):
    legal_moves = []
    dirs = [[-2, -1], [-2, 1], [-1, -2], [-1, 2],
        [1, -2], [1, 2], [2, -1], [2, 1]]
    opp_str = 'b' if knight_str[0] == 'w' else 'w'

    for dir in dirs:
      row = i + dir[0]
      col = j + dir[1]
      if row >= 0 and col >= 0 and row < 4 and col < 4:
        if self.board[row, col] == '.' or self.board[row, col][0] == opp_str:
          legal_moves.append(action_space[knight_str+'->('+str(row)+','+str(col)+')'])
    return legal_moves

  def _legal_king_moves(self, i, j, king_str):
    legal_moves = []
    dirs = [[-1, -1], [-1, 1], [1, -1], [1, 1],
        [-1, 0], [1, 0], [0, -1], [0, 1]]
    opp_str = 'b' if king_str[0] == 'w' else 'w'

    for dir in dirs:
      row = i + dir[0]
      col = j + dir[1]
      if row >= 0 and col >= 0 and row < 4 and col < 4:
        if self.board[row, col] == '.' or self.board[row, col][0] == opp_str:
          legal_moves.append(action_space[king_str+'->('+str(row)+','+str(col)+')'])
    return legal_moves

  def _apply_action(self, action):
    """Applies the specified action to the state.
    This function essentially maps the integer action to the corresponding effect it should
    have on the state of the game. This is done by finding the corresponding piece and movement
    of the piece on the board and then search for where the piece initiatially was and updating the 
    board accordingly. The current player is then updated and the game is checked to see if it is over.
    This is easily done by checking if there is only one king left on the board.
    
    
    Complexity wise this function is O(m*n) where m is the number of rows and n is the number of columns.
    This is because the function loops through the board to find the piece that is being moved and then
    loops through the board again to find the piece that is being captured. The function could be made
    more efficient by keeping track of the location of the pieces and then updating the board accordingly
    but this would require more memory and would be more complicated to implement.
    """
    
    action_str = space_action[action]
    print(f'action_str: {action_str}')
    piece = action_str.split('->')[0]
    i,j = int(action_str[-4]), int(action_str[-2])
    
    for k in range(4):
      for l in range(4):
        if self.board[k, l] == piece:
          self.board[k, l] = '.'
          break
    self.board[i, j] = piece
    
    K_count = 0
    for i in range(4):
      for j in range(4):
        if len(self.board[i, j])>1 and self.board[i, j][1] == 'K':
          K_count += 1
    if K_count == 1:
      self._is_terminal = True
      self._player0_score = 1.0 if self._cur_player == 0 else -1.0
    else:
      self._cur_player = 1 - self._cur_player

  def _action_to_string(self, player, action):
    """Action -> string.
    This just gets the string representation of the action.
    """
    return space_action[action]

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.
    '''
    There are 13 possible pieces that can be on the board with the 4 separate pawns, 8 special
    pieces and empty spaces. The observation is a 1-D tensor of size 208 which is 13*4*4. The
    '''
    self.tensor = np.zeros(208, np.float32)
    self.dict = {"observation": np.reshape(self.tensor, (13, 4, 4))}
    self.piece_to_index = {'bP1':0, 'bP2':1, 'bB':2, 'bR':3, 'bN':4, 'bK':5,
                           'wP1':6, 'wP2':7, 'wB':8, 'wR':9, 'wN':10, 'wK':11,
                           '.':12}
  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs.fill(0)
    
    for i in range(4):
      for j in range(4):
        piece = state.board[i, j]
        obs[self.piece_to_index[piece], i, j] = 1
    

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)


def _board_to_string(board):
  """Returns a string representation of the board."""
  return "\n".join("".join(row) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, MiniChessGame)

if __name__=='__main__':
    game = pyspiel.load_game('python_mini_chess')
    
    debug_main(game) # This runs one completely random instance of the game
    # debug_alpha_zero(game) # This trains an alpha zero agent on the game and save the checkpoints to mini_chess_alpha_zero
    debug_mcts_evaluator([], game=game, az_path='open_spiel/python/games/mini_chess_alpha_zero/checkpoint-475')
    # This runs a Monte Carlo Tree Search agent on the game using the alpha zero agent as the evaluator
    



