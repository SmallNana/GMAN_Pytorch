import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data

parser = argparse.ArgumentParser()

parser.add_argument('--time_slot', type=int, default=5, help='一个时间步为5mins')
parser.add_argument('--num_his', type=int, default=12, help='历史时间步长度')
parser.add_argument('--num_pred', type=int, default=12, help='预测时间步长度')
parser.add_argument('--L', type=int, default=3, help='时空块数量')
parser.add_argument('--K', type=int, default=8, help='Mulit-heads数量')
parser.add_argument('--d', type=int, default=8, help='单head维度')
parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集数量比例')
parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集数量比例')
parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集数量比例')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='跑几代')
parser.add_argument('--patience', type=int, default=10, help='等待代数')
parser.add_argument('--learning_rate', type=float, default=0.001, help='初始学习率')
parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')
parser.add_argument('--traffic_file', default='./data/pems-bay.h5', help='traffic file')
parser.add_argument('--SE_file', default='./data/SE(PeMS).txt', help='spatial embedding file')
parser.add_argument('--model_file', default='./data/GMAN.pkl', help='模型保存路径')
parser.add_argument('--log_file', default='./data/log', help='log file')

args = parser.parse_args()

(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(args)

num_sampes, num_steps, num_sensors = testY.shape

testPred = np.loadtxt('./figure/testPred.txt').reshape(num_sampes, num_steps, num_sensors)
testY = np.loadtxt('./figure/testY.txt').reshape(num_sampes, num_steps, num_sensors)
print('加载数据完成')


plt.figure(figsize=(10, 280))
for k in tqdm.tqdm(range(325)):
    # 325个sensors
    plt.subplot(325, 1, k + 1)
    # 第k+1个子图
    for j in range(len(testPred)):
        # 样本数量
        c, d = [], []
        for i in range(12):
            # 时间 1个小时
            c.append(testPred[j, i, k])
            d.append(testY[j, i, k])
        plt.plot(range(1 + j, 12 + 1 + j), c, c='b')
        # 在x轴的[1+j,12+1+j)上绘制车速
        plt.plot(range(1 + j, 12 + 1 + j), d, c='r')

plt.title('Test prediction vs Target')
plt.savefig('./figure/test_results.png')