import math
import time

import argparse
import numpy as np
import torch
import tqdm
import datetime

from utils import log_string
from utils import load_data
from utils import metric

device = torch.device('cuda')


def test(args, log):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)

    num_train, _, num_vertex = trainX.shape  # [num_samples, his_steps, num_vertix]
    num_val = valX.shape[0]
    num_test = testX.shape[0]

    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)

    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % args.model_file)
    model = torch.load(args.model_file).to(device)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    with torch.no_grad():
        model.eval()
        trainPred = []

        for batch_idx in tqdm.tqdm(range(train_num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx].to(device)
            TE = trainTE[start_idx: end_idx].to(device)
            pred_batch = model(X, TE)

            trainPred.append(pred_batch.detach().cpu().clone())
            del X, TE, pred_batch

        trainPred = torch.from_numpy(np.concatenate(trainPred,axis=0))
        trainPred = trainPred * std + mean

        valPred = []
        for batch_idx in tqdm.tqdm(range(val_num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            X = valX[start_idx: end_idx].to(device)
            TE = valTE[start_idx: end_idx].to(device)
            pred_batch = model(X, TE)

            valPred.append(pred_batch.detach().cpu().clone())
            del X, TE, pred_batch

        valPred = torch.from_numpy(np.concatenate(valPred, axis=0))
        valPred = valPred * std + mean

        testPred = []
        start_test = time.time()

        for batch_idx in tqdm.tqdm(range(test_num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
            X = testX[start_idx: end_idx].to(device)
            TE = testTE[start_idx: end_idx].to(device)
            pred_batch = model(X, TE)
            testPred.append(pred_batch.detach().cpu().clone())
            del X, TE, pred_batch
        testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
        testPred = testPred * std + mean

    end_test = time.time()
    train_mae, train_rmse, train_mape = metric(trainPred, trainY)
    val_mae, val_rmse, val_mape = metric(valPred, valY)
    test_mae, test_rmse, test_mape = metric(testPred, testY)

    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae, train_rmse, train_mape * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae, val_rmse, val_mape * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))
    log_string(log, 'performance in each prediction step')

    MAE, RMSE, MAPE = [], [], []
    for step in range(args.num_pred):
        mae, rmse, mape = metric(testPred[:, step], testY[:, step])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                   (step + 1, mae, rmse, mape * 100))

    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    log_string(
        log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (average_mae, average_rmse, average_mape * 100))

    return trainPred, valPred, testPred



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--time_slot', type=int, default=5, help='一个时间步为5mins')
    parser.add_argument('--num_his', type=int, default=12, help='历史时间步长度')
    parser.add_argument('--num_pred', type=int, default=12, help='预测时间步长度')
    parser.add_argument('--L', type=int, default=5, help='时空块数量')
    parser.add_argument('--K', type=int, default=8, help='Mulit-heads数量')
    parser.add_argument('--d', type=int, default=8, help='单head维度')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集数量比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集数量比例')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集数量比例')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=1000, help='跑几代')
    parser.add_argument('--patience', type=int, default=10, help='等待代数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')
    parser.add_argument('--traffic_file', default='./data/metr-la.h5', help='traffic file')
    parser.add_argument('--SE_file', default='./data/SE(METR).txt', help='spatial embedding file')
    parser.add_argument('--model_file', default='./data/GMAN.pkl', help='模型保存路径')
    parser.add_argument('--log_file', default='./data/log', help='log file')

    args = parser.parse_args()
    log = open(args.log_file, 'w')
    log_string(log, str(args)[10: -1])
    trainPred, valPred, testPred = test(args, log)

    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)


    trainPred_ = trainPred.numpy().reshape(-1, trainY.shape[-1])
    trainY_ = trainY.numpy().reshape(-1, trainY.shape[-1])
    valPred_ = valPred.numpy().reshape(-1, valY.shape[-1])
    valY_ = valY.numpy().reshape(-1, valY.shape[-1])
    testPred_ = testPred.numpy().reshape(-1, testY.shape[-1])
    testY_ = testY.numpy().reshape(-1, testY.shape[-1])

    # 保存数据
    l = [trainPred_, trainY_, valPred_, valY_, testPred_, testY_]
    name = ['trainPred', 'trainY', 'valPred', 'valY', 'testPred', 'testY']
    for i, data in tqdm.tqdm(enumerate(l)):
        np.savetxt('./figure/' + name[i] + '.txt', data, fmt='%s')





