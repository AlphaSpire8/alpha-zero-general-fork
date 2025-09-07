import logging
import coloredlogs

from Coach import Coach
# 1. --- 导入五子棋的游戏和我们刚创建的PyTorch网络 ---
from gobang.GobangGame import GobangGame as Game
from gobang.pytorch.NNet import NNetWrapper as nn

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# 2. --- 为五子棋调整参数 ---
# 五子棋更复杂，需要更多的训练、对弈和思考深度
args = dotdict({
    'numIters': 20,             # 总迭代次数
    'numEps': 60,              # 每次迭代的自我对弈局数
    'tempThreshold': 15,
    'updateThreshold': 0.6,    # 新模型接受阈值
    'maxlenOfQueue': 200000,    # 经验队列长度
    'numMCTSSims': 60,          # MCTS模拟次数 (比井字棋要多)
    'arenaCompare': 40,         # 新旧模型对比局数
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info('Loading %s...', Game.__name__)
    # 3. --- 初始化五子棋游戏 ---
    # GobangGame接收一个棋盘尺寸n作为参数。
    # 标准是15x15，但为了训练更快，我们可以从一个稍小的棋盘开始，比如 9x9
    g = Game(15)

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