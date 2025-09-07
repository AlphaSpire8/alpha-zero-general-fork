import Arena
from MCTS import MCTS
# 1. --- 导入井字棋的游戏和玩家 ---
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
# 2. --- 导入我们为井字棋创建的PyTorch神经网络 ---
from tictactoe.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

# 3. --- 设置为人类 vs AI 模式 ---
human_vs_cpu = True

# 4. --- 初始化井字棋游戏 ---
g = TicTacToeGame()

# 5. --- 定义井字棋的玩家 ---
# 人类玩家
hp = HumanTicTacToePlayer(g).play
# 随机玩家 (可以用来测试)
rp = RandomPlayer(g).play


# 6. --- 加载你刚刚训练好的AI模型 ---
# 创建神经网络
n1 = NNet(g)
# 从你的训练输出文件夹中加载模型权重
# 训练脚本会将最好的模型保存为 'best.pth.tar'
n1.load_checkpoint('./temp/','best.pth.tar')

# 7. --- 配置MCTS (蒙特卡洛树搜索) ---
# numMCTSSims 决定了AI思考的深度，值越高棋力越强，但反应越慢
# 对于井字棋，25次模拟已经足够了
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
# 定义AI玩家的执行函数：使用MCTS模拟后，选择概率最高的一步
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


# 8. --- 设置对手 ---
if human_vs_cpu:
    player2 = hp  # 对手是人类
else:
    # 如果你想让AI和自己对战，可以取消下面的注释
    # n2 = NNet(g)
    # n2.load_checkpoint('./temp/', 'best.pth.tar')
    # args2 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
    # mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    # player2 = n2p
    player2 = rp # 或者让AI和随机玩家对战


# 9. --- 创建竞技场并开始游戏 ---
# Arena的第一个参数是player1 (我们的AI)，第二个是player2 (人类或另一个AI)
# display函数需要使用井字棋的display函数
arena = Arena.Arena(n1p, player2, g, display=TicTacToeGame.display)

# 10. --- 开始对战！ ---
# playGames的第一个参数是对战的局数
print("Starting the game! You are Player 2.")
print(arena.playGames(1, verbose=True))