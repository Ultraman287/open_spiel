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
"""Tic tac toe (noughts and crosses), implemented in Python.

This is a demonstration of implementing a deterministic perfect-information
game in Python.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python-implemented games. This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that (e.g. CFR algorithms). It is likely to be poor if the algorithm
relies on processing and updating states as it goes, e.g., MCTS.
"""

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
from mini_chess_helper import action_space, space_action, debug_main, debug_alpha_zero, debug_mcts_evaluator
import pyspiel
import numpy as np
import subprocess


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
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=28+16+32+32+32,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=50)


class MiniChessGame(pyspiel.Game):
  """A Python version of Mini Chess."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return MiniChessState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class MiniChessState(pyspiel.State):
  """A python version of the Tic-Tac-Toe state."""

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
    """Applies the specified action to the state."""
    
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
    """Action -> string."""
    return space_action[action]
    # row, col = _coord(action)
    # return "{}({},{})".format("x" if player == 0 else "o", row, col)

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
    
    debug_main(game)
    debug_mcts_evaluator([], game=game, az_path='open_spiel/python/games/mini_chess_alpha_zero/checkpoint-475')
    # debug_alpha_zero(game)
    # state = game.new_initial_state()
    # print(state)
    # while not state.is_terminal():
    #     print(state.legal_actions())
    #     action = int(input())
    #     state.apply_action(action)
    #     print(state)
    # print(state.returns())
  # I want to use examples/example.py to run this game


