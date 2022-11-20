from __future__ import print_function
import pickle
import random
from AlphaZeroAgent import AlphaZeroAgent
from HumanAgent import HumanAgent
from MCTSAgent import MCTSAgent
from game import Board, Game

if __name__ == '__main__':
    model_file = 'models/best_policy_664_1.pt'

    board = Board(width=6, height=6, n_in_row=4)
    simulator = Game(board)

    alphazeroPlayer = AlphaZeroAgent(board, model_file=model_file, n_playouts=400)
    pureMCTS = MCTSAgent(n_playout=1000)

    human = HumanAgent()

    simulator.start_play(human, alphazeroPlayer, start_player=random.choice([0, 1]), is_shown=1)
