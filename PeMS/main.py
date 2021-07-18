import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import log_string, plot_train_val_loss
from utils import count_parameters, load_data

from model import GMAN
from train import train
from test import test

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
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])

T = 24 * 60 // args.time_slot  # Number of time steps in one day


# load data
log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
 testY, SE, mean, std) = load_data(args)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')

del trainX, trainTE, valX, valTE, testX, testTE, mean, std

device = torch.device('cuda')
# 建立模型
log_string(log, 'compiling model...')
SE = SE.to(device)

model = GMAN(SE, args, bn_decay=0.1).to(device)
loss_criterion = nn.L1Loss().to(device)

optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=0.9)

parameters = count_parameters(model)
log_string(log, 'trainable parameters: {:,}'.format(parameters))
log_string(log, 'GPU使用情况:{:,}'.format(torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0))

if __name__ == '__main__':
    start = time.time()
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)
    trainPred, valPred, testPred = test(args, log)
    end = time.time()

    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()





