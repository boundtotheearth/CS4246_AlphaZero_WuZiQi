# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
Obtained From: https://github.com/junxiaosong/AlphaZero_Gomoku
"""

from MCTS import MCTS
import numpy as np


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), None

class MCTSAgent(object):
    def __init__(self, n_playout=2000):
        self.n_playout = n_playout
        self.mcts = MCTS(policy_value_fn, n_playout=n_playout)

    def set_player_ind(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")
    
    def __str__(self):
        return "Player {}: MCTS".format(self.player)