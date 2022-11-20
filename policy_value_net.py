import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyValueNet(nn.Module):
    def __init__(self, board):
        super(PolicyValueNet, self).__init__()
        self.board_width = board.width
        self.board_height = board.height

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * self.board_width * self.board_height, self.board_width * self.board_height)

        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * self.board_width * self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-4)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x_act = self.act_conv1(x)
        x_act = F.relu(x_act)
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act)

        x_val = self.val_conv1(x)
        x_val = F.relu(x_val)
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = self.val_fc1(x_val)
        x_val = F.relu(x_val)
        x_val = self.val_fc2(x_val)
        x_val = torch.tanh(x_val)

        return x_act, x_val