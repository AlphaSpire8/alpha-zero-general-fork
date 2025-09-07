import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# 从项目中获取超参数
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(), # 自动检测CUDA
    'num_channels': 512,
})

class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(TicTacToeNNet, self).__init__()
        
        # 将3x3的棋盘展平为1x9的向量
        self.fc1 = nn.Linear(self.board_x * self.board_y, args.num_channels)
        self.fc2 = nn.Linear(args.num_channels, args.num_channels)

        # 输出两个头：策略头和价值头
        self.fc3 = nn.Linear(args.num_channels, self.action_size) # 策略头
        self.fc4 = nn.Linear(args.num_channels, 1) # 价值头

        self.bn1 = nn.BatchNorm1d(args.num_channels)
        self.bn2 = nn.BatchNorm1d(args.num_channels)

    def forward(self, s):
        # s: batch_size x board_x x board_y
        # 将输入展平
        s = s.view(-1, self.board_x * self.board_y) 
        
        # 通过全连接层和激活函数
        x = F.dropout(F.relu(self.bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), p=self.args.dropout, training=self.training)
        
        # 策略头 (Policy Head)
        pi = self.fc3(x)
        
        # 价值头 (Value Head)
        v = self.fc4(x)

        # 返回策略的log_softmax和价值的tanh
        return F.log_softmax(pi, dim=1), torch.tanh(v)