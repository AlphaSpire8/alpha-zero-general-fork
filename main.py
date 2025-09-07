import logging
import coloredlogs

from Coach import Coach
# 1. 修改导入的游戏类
# from othello.OthelloGame import OthelloGame as Game  <- 注释掉或删除这一行
from tictactoe.TicTacToeGame import TicTacToeGame as Game # <- 添加这一行

# 2. 修改导入的神经网络包装器
from tictactoe.pytorch.NNet import NNetWrapper as nn 

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# 3. 调整参数以适应井字棋
args = dotdict({
    'numIters': 20,             # 井字棋简单，不需要1000次迭代
    'numEps': 20,               # 每次迭代的自我对弈局数，可以适当减少
    'tempThreshold': 5,
    'updateThreshold': 0.6,    # 接受新模型的胜率阈值，可以稍微降低
    'maxlenOfQueue': 20000,    # 经验队列的最大长度
    'numMCTSSims': 15,          # MCTS模拟次数
    'arenaCompare': 30,         # 新旧模型对比的局数
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,        # 第一次训练时设为False
    'load_folder_file': ('./temp/','best.pth.tar'), # 如果要加载模型，确保路径正确
    'numItersForTrainExamplesHistory': 30,
})


def main():
    # 4. 修改游戏初始化
    log.info('Loading %s...', Game.__name__)
    # g = Game(6) # <- 这是奥赛罗的初始化方式，带棋盘大小参数
    g = Game()    # <- 井字棋的初始化不需要参数

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()