import math
import time

import numpy as np
import torch
import tqdm
import datetime

from utils import plot_train_val_loss
from utils import log_string
from utils import load_data
from model import *

device = torch.device('cuda')


def train(model, args, log, loss_criterion, optimizer, scheduler):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std) = load_data(args)

    num_train, _, num_vertex = trainX.shape  # [num_samples, his_steps, num_vertix]
    log_string(log, '**** training model ****')
    num_val = valX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)

    wait = 0
    val_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []

    trainX.to(device)
    trainY.to(device)
    trainTE.to(device)

    model.to(device)

    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break

        # 洗牌
        permutation = torch.randperm(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]

        # 训练
        start_train = time.time()
        model.train()
        train_loss = 0
        for batch_idx in tqdm.tqdm(range(train_num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx].to(device)
            TE = trainTE[start_idx: end_idx].to(device)
            label = trainY[start_idx: end_idx].to(device)

            optimizer.zero_grad()
            pred = model(X, TE)
            pred = pred * std + mean

            loss_batch = loss_criterion(pred, label)
            train_loss += float(loss_batch.detach().cpu().item()) * (end_idx - start_idx)
            loss_batch.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (batch_idx+1) % 5 == 0:
                print(f'Training batch: {batch_idx + 1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')

            del X, TE, label, pred, loss_batch

        train_loss /= num_train
        train_total_loss.append(train_loss)
        end_train = time.time()

        # 验证
        start_val = time.time()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx: end_idx].to(device)
                TE = valTE[start_idx: end_idx].to(device)
                label = valY[start_idx: end_idx].to(device)

                pred = model(X, TE)
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += float(loss_batch.detach().cpu().item()) * (end_idx - start_idx)

                del X, TE, label, pred, loss_batch

        val_loss /= num_val
        val_total_loss.append(val_loss)
        end_val = time.time()

        log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
             args.max_epoch, end_train - start_train, end_val - start_val)
        )

        log_string(
            log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}'
            )
            wait = 0
            val_loss_min = val_loss
            best_model_wts = model.state_dict()
        else:
            wait += 1
        np.save('./train_loss', train_total_loss)
        np.save('./val_loss', val_total_loss)
        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_file)
    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')

    return train_total_loss, val_total_loss






