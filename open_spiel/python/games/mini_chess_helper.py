import random
from absl import app
from absl import flags
import numpy as np

'''
This is a helper file designed to implement the required tweaks to the existing implementation
of open spiel in order to accomodate the mini chess game. This file is not meant to be run
'''

from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import spawn
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import collections
import random
import sys
import pyspiel

# We need to change some of the functions within the library itself to make this
# implementation work.
# open_spiel/python/algorithms/alpha_zero/alpha_zero.py
# 498-501
# def alpha_zero(config: Config, game: pyspiel.Game = None):
#   """Start all the worker processes for a full alphazero setup."""
#   if game is None:
#     game = pyspiel.load_game(config.game)

# Similarly, for mcts.py, we need to change the following:


'''
The action space for mini chess was designed as follows:
    1. The Pawns and Bishops are first mapped to all the squares that they can move to manually
    since they have a different action space than the rest of the pieces
    2. The rest of the pieces are iteratively mapped to all the squares within the baord and added to
    the action space
'''
action_space = {}
action_space['wP1->(1,0)'] = 0
action_space['wP1->(1,1)'] = 1
action_space['wP1->(1,2)'] = 2
action_space['wP1->(0,0)'] = 3
action_space['wP1->(0,1)'] = 4
action_space['wP1->(0,2)'] = 5
action_space['wP1->(0,3)'] = 6
action_space['wP2->(1,1)'] = 7
action_space['wP2->(1,2)'] = 8
action_space['wP2->(1,3)'] = 9
action_space['wP2->(0,0)'] = 10
action_space['wP2->(0,1)'] = 11
action_space['wP2->(0,2)'] = 12
action_space['wP2->(0,3)'] = 13
action_space['bP1->(2,0)'] = 14
action_space['bP1->(2,1)'] = 15
action_space['bP1->(2,2)'] = 16
action_space['bP1->(3,0)'] = 17
action_space['bP1->(3,1)'] = 18
action_space['bP1->(3,2)'] = 19
action_space['bP1->(3,3)'] = 20
action_space['bP2->(2,1)'] = 21
action_space['bP2->(2,2)'] = 22
action_space['bP2->(2,3)'] = 23
action_space['bP2->(3,0)'] = 24
action_space['bP2->(3,1)'] = 25
action_space['bP2->(3,2)'] = 26
action_space['bP2->(3,3)'] = 27
action_space['wB->(3,0)'] = 28
action_space['wB->(2,1)'] = 29
action_space['wB->(1,2)'] = 30
action_space['wB->(0,3)'] = 31
action_space['wB->(0,1)'] = 32
action_space['wB->(2,3)'] = 33
action_space['wB->(1,0)'] = 34
action_space['wB->(3,2)'] = 35
action_space['bB->(0,0)'] = 36
action_space['bB->(1,1)'] = 37
action_space['bB->(2,2)'] = 38
action_space['bB->(3,3)'] = 39
action_space['bB->(1,3)'] = 40
action_space['bB->(3,1)'] = 41
action_space['bB->(0,2)'] = 42
action_space['bB->(2,0)'] = 43

cur = 44
go_over_all = ['wR','bR','wN','bN','wK','bK']

for piece in go_over_all:
    for i in range(4):
        for j in range(4):
            action_space[piece+'->('+str(i)+','+str(j)+')'] = cur
            cur += 1

'''
This keeps a specific edge case in mind where the pawn's initial position, even if not a
possible action, is still the last action that ocurred for the action space to get to the 
initial state which means it won't get called separately.
'''
action_space['wP1->(2,1)'] = 140
action_space['wP2->(2,2)'] = 141
action_space['bP1->(1,1)'] = 142
action_space['bP2->(1,2)'] = 143
space_action = {v: k for k, v in action_space.items()}

def debug_main(game: pyspiel.Game):
  '''
  This is an altered version of the main function in the open_spiel/python/examples/example.py filled
  with all the completed flags that codify the mini chess game within the context of open spiel. 
  '''
    
  state = game.new_initial_state()
  
  # Print the initial state
  print(str(state))

  while not state.is_terminal():
      # This just successively plays random actions until the game is done.
      action = random.choice(state.legal_actions(state.current_player()))
      action_string = state.action_to_string(state.current_player(), action)
      print("Player ", state.current_player(), ", randomly sampled action: ",
            action_string)
      state.apply_action(action)

  print(str(state))

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


'''
Below is an altered version of the examples/alpha_zero.py file that is filled with all the
completed flags for the hyperparameters that are going to be used for the mini chess game.

Since the default values were using estimates for the original alphazero paper, something built on much more
data with a much larger action space, I had to change the values to be more appropriate for the mini chess game.
'''
default_values = {
    "game": "mini_chess",
    "uct_c": 2,
    "max_simulations": 20,
    "train_batch_size": 2 ** 5,
    "replay_buffer_size": 2 ** 8,
    "replay_buffer_reuse": 3,
    "learning_rate": 0.1,
    "weight_decay": 0.0001,
    "policy_epsilon": 0.25,
    "policy_alpha": 1,
    "temperature": 1,
    "temperature_drop": 10,
    "nn_model": "mlp",
    "nn_width": 16,
    "nn_depth": 5,
    "path": "./mini_chess_alpha_zero/",
    "checkpoint_freq": 25,
    "actors": 2,
    "evaluators": 1,
    "evaluation_window": 25,
    "eval_levels": 7,
    "max_steps": 0,
    "quiet": True,
    "verbose": False
}



def debug_alpha_zero(game: pyspiel.Game):
    config = alpha_zero.Config(
        game=default_values["game"],
        path=default_values["path"],
        learning_rate=default_values["learning_rate"],
        weight_decay=default_values["weight_decay"],
        train_batch_size=default_values["train_batch_size"],
        replay_buffer_size=default_values["replay_buffer_size"],
        replay_buffer_reuse=default_values["replay_buffer_reuse"],
        max_steps=default_values["max_steps"],
        checkpoint_freq=default_values["checkpoint_freq"],
        actors=default_values["actors"],
        evaluators=default_values["evaluators"],
        uct_c=default_values["uct_c"],
        max_simulations=default_values["max_simulations"],
        policy_alpha=default_values["policy_alpha"],
        policy_epsilon=default_values["policy_epsilon"],
        temperature=default_values["temperature"],
        temperature_drop=default_values["temperature_drop"],
        evaluation_window=default_values["evaluation_window"],
        nn_model=default_values["nn_model"],
        nn_width=default_values["nn_width"],
        nn_depth=default_values["nn_depth"],
        eval_levels=default_values["eval_levels"],
        observation_shape=None,
        output_size=None,
        quiet=default_values["quiet"],
    )
    alpha_zero.alpha_zero(config, game=game) # this uses open_spiel's built in alpha zero algorithm slightly modified
    # to allow for the game instance to be inputted directly instead of having to build it in separately into the library
    


'''
This is an altered version of the examples/mcts.py file that is filled with all the
completed flags for the required information for the mcts agent to run on the mini chess game.

In this case, it uses the az_model which is essentially a checkpoint created through prior training with
the alpha zero algorithm. The actuall implementation has been abstracted well enough that just by running this
you can play against a well-trained mcts agent on this tiny board from the command line.
'''

mcts_flags = {
    "game": "tic_tac_toe",
    "player1": "human",
    "player2": "az",
    "gtp_path": None,
    "gtp_cmd": [],
    "az_path": './open_spiel/python/games/mini_chess_alpha_zero/',
    "uct_c": 2,
    "rollout_count": 1,
    "max_simulations": 1000,
    "num_games": 1,
    "seed": None,
    "random_first": False,
    "solve": True,
    "quiet": False,
    "verbose": False,
}
    
    
def _opt_print(*args, **kwargs):
    if not mcts_flags["quiet"]:
        print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(mcts_flags["seed"])
    if bot_type == "mcts":
        evaluator = mcts.RandomRolloutEvaluator(mcts_flags["rollout_count"], rng)
        return mcts.MCTSBot(
                game,
                mcts_flags["uct_c"],
                mcts_flags["max_simulations"],
                evaluator,
                random_state=rng,
                solve=mcts_flags["solve"],
                verbose=mcts_flags["verbose"])
    if bot_type == "az":
        model = az_model.Model.from_checkpoint(mcts_flags["az_path"])
        evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
        return mcts.MCTSBot(
                game,
                mcts_flags["uct_c"],
                mcts_flags["max_simulations"],
                evaluator,
                random_state=rng,
                child_selection_fn=mcts.SearchNode.puct_value,
                solve=mcts_flags["solve"],
                verbose=mcts_flags["verbose"])
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    if bot_type == "gtp":
        bot = gtp.GTPBot(game, mcts_flags["gtp_path"])
        for cmd in mcts_flags["gtp_cmd"]:
            bot.gtp_cmd(cmd)
        return bot
    raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action(state, action_str):
    for action in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    return None


def _play_game(game, bots, initial_actions):
    """Plays one game."""
    state = game.new_initial_state()
    _opt_print("Initial state:\n{}".format(state))

    history = []

    if mcts_flags["random_first"]:
        assert not initial_actions
        initial_actions = [state.action_to_string(
                state.current_player(), random.choice(state.legal_actions()))]

    for action_str in initial_actions:
        action = _get_action(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))

        history.append(action_str)
        for bot in bots:
            bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)
        _opt_print("Forced action", action_str)
        _opt_print("Next state:\n{}".format(state))

    while not state.is_terminal():
        current_player = state.current_player()
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            _opt_print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            action_str = state.action_to_string(current_player, action)
            _opt_print("Sampled action: ", action_str)
        elif state.is_simultaneous_node():
            raise ValueError("Game cannot have simultaneous nodes.")
        else:
            # Decision node: sample action for the single current player
            bot = bots[current_player]
            action = bot.step(state)
            action_str = state.action_to_string(current_player, action)
            _opt_print("Player {} sampled action: {}".format(current_player,
                                                                                                             action_str))

        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
        history.append(action_str)
        state.apply_action(action)

        _opt_print("Next state:\n{}".format(state))

    # Game is now done. Print return for each player
    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)), ", Game actions:",
                " ".join(history))

    for bot in bots:
        bot.restart()

    return returns, history


def debug_mcts_evaluator(argv, game: pyspiel.Game = None, **kwargs):
    for k, v in kwargs.items(): # this is to allow for dynamically changing the flags from where the files getting run
        mcts_flags[k] = v
    if game == None:
        game = pyspiel.load_game(mcts_flags["game"])
    if game.num_players() > 2:
        sys.exit("This game requires more players than the example can handle.")
    bots = [
            _init_bot(mcts_flags["player1"], game, 0),
            _init_bot(mcts_flags["player2"], game, 1),
    ]
    histories = collections.defaultdict(int)
    overall_returns = [0, 0]
    overall_wins = [0, 0]
    game_num = 0
    try:
        for game_num in range(mcts_flags["num_games"]):
            returns, history = _play_game(game, bots, argv[1:])
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
    except (KeyboardInterrupt, EOFError):
        game_num -= 1
        print("Caught a KeyboardInterrupt, stopping early.")
    print("Number of games played:", game_num + 1)
    print("Number of distinct games played:", len(histories))
    print("Players:", mcts_flags["player1"], mcts_flags["player2"])
    print("Overall wins", overall_wins)
    print("Overall returns", overall_returns)
