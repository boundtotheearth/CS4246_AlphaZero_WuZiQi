from MCTS import MCTS
from policy_value_net import PolicyValueNet
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class AlphaZeroAgent(object):
    def __init__(self, board, n_playouts=2000, is_selfplay=0, lr=2e-3, kl_targ=0.02, model_file=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board_width = board.width
        self.board_height = board.height
        self.n_playout = n_playouts
        self.policy_value_net = PolicyValueNet(board).to(self.device)
        self.mcts = MCTS(self.policy_value_fn, n_playout=n_playouts)
        self.is_selfplay = is_selfplay
        self.lr = lr
        self.lr_multiplier = 1.0
        self.kl_targ = kl_targ

        if model_file != None:
            model_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(model_params)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)
    
    def get_action(self, board, return_prob=0, temp=1e-03):
        if len(board.availables) <= 0:
            print("BOARD IS FULL")
            return None
        move_probs = np.zeros(board.width * board.height)
        actions, probs = self.mcts.get_move_probs(board, temp=temp)
        move_probs[list(actions)] = probs
        move = None
        if self.is_selfplay:
            move = np.random.choice(actions, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(actions, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move

    def policy_value_fn(self, board):
        with torch.no_grad():
            legal_actions = list(board.availables)
            current_state = board.current_state().to(self.device)
            log_probs, value = self.policy_value_net(current_state)
            act_probs = torch.exp(log_probs.flatten())
            act_probs = zip(legal_actions, act_probs[legal_actions])
            value = value.item()
            return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch):
        state_batch = torch.stack(state_batch).to(self.device)
        mcts_probs = torch.tensor(np.array(mcts_probs), dtype=torch.float, device=self.device)
        winner_batch = torch.tensor(np.array(winner_batch), dtype=torch.float, device=self.device)
    
        self.policy_value_net.optimizer.zero_grad()
        
        old_log_probs, old_value = self.policy_value_net(state_batch)

        value_loss = F.mse_loss(old_value.view(-1), winner_batch)
        # policy_loss = -torch.mean(torch.sum(mcts_probs*old_log_probs, 1))
        policy_loss = F.cross_entropy(old_log_probs, mcts_probs)
        loss = value_loss + policy_loss
        loss.backward()
        self.policy_value_net.optimizer.step()

        new_log_probs, new_value = self.policy_value_net(state_batch)
        new_probs = torch.exp(new_log_probs)
        entropy = -torch.mean(torch.sum(new_probs * new_log_probs, 1)).item()
        kl_div = F.kl_div(old_log_probs, new_probs, reduction='batchmean').item()

        if kl_div > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl_div < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        
        for param_group in self.policy_value_net.optimizer.param_groups:
            param_group['lr'] = self.lr * self.lr_multiplier

        return loss.item(), entropy, kl_div, self.lr_multiplier
        
    def save_model(self, path):
        torch.save(self.policy_value_net.state_dict(), path)

    def __str__(self):
        return "Player {}: AlphaZero".format(self.player)

