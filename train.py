import os
import sys

import json
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from SGHMC_Bayesian import sghmc
import util


def train(X_train, y_train, X_valid, y_valid, X_test, y_test, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, last_step = 0, 0
    model.train()
    batch = args.batch_size
    for epoch in range(1, args.epochs+1):
        corrects = 0
        for idx in range(int(X_train.shape[0]/batch) + 1):
            feature = torch.LongTensor(X_train[(idx*batch):(idx*batch+batch),])
            target = torch.LongTensor(y_train[(idx*batch):(idx*batch+batch)])
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            #sghmc(logit, target, model, eta=0.00002, L=5, alpha=0.01, V=1)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum().item()
            accuracy = 100.0 * corrects / batch / (idx + 1)
            sys.stdout.write('\rEpoch[{}] Batch[{}] - loss: {:.4f}  acc: {:.2f}%({}/{})'.format(
                             epoch, idx, loss.item(), accuracy, corrects, batch * (idx + 1)))
        print('\n')
        eval(X_valid, y_valid, model, "Validation", args)
        dev_acc = eval(X_test, y_test, model, "Testing   ", args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            last_step = idx
            if args.save_best:
                save(model, args.save_dir, 'best', epoch)
        save(model, args.save_dir, 'snapshot', epoch)


def eval(X, y, model, term, args):
    model.eval()
    corrects, avg_loss = 0, 0
    correct_part, total_part = {0.1:0, 0.2:0, 0.3:0, 0.4:0}, {0.1:1e-16, 0.2:1e-16, 0.3:1e-16, 0.4:1e-16}
    batch = args.batch_size

    for idx in range(int(X.shape[0]/batch) + 1):
        feature = torch.LongTensor(X[(idx*batch):(idx*batch+batch),])
        target = torch.LongTensor(y[(idx*batch):(idx*batch+batch)])
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data.item()
        predictor = torch.exp(logit[:, 1]) / (torch.exp(logit[:, 0]) + torch.exp(logit[:, 1]))
        for xnum in range(1, 5):
            thres = round(0.1 * xnum, 1)
            idx_thres = (predictor > 0.5 + thres) + (predictor < 0.5 - thres)
            correct_part[thres] += (torch.max(logit, 1)[1][idx_thres] == target.data[idx_thres]).sum().item()
            total_part[thres] += idx_thres.sum().item()

        corrects += (torch.max(logit, 1)[1] == target.data).sum().item()

    size = y.shape[0]
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('         {} - loss: {:.4f}  acc: {:.2f}%({}/{}) {:.2f}%({}/{}) {:.2f}%({}/{}) {:.2f}%({}/{}) {:.2f}%({}/{}) \n'.format(term,
          avg_loss, accuracy, corrects, size, 100.0 * correct_part[0.1] / total_part[0.1], correct_part[0.1], int(total_part[0.1]), 
          100.0 * correct_part[0.2] / total_part[0.2], correct_part[0.2], int(total_part[0.2]), 100.0 * correct_part[0.3] / total_part[0.3], 
          correct_part[0.3], int(total_part[0.3]), 100.0 * correct_part[0.4] / total_part[0.4], correct_part[0.4], int(total_part[0.4])))
    return accuracy


def predict(model, feature, sen_len, word2idx, stopWords, cuda_flag):
    model.eval()
    if cuda_flag:
        feature = feature.cuda()
    logit = model(feature)
    predictor = torch.exp(logit[:, 1]) / (torch.exp(logit[:, 0]) + torch.exp(logit[:, 1]))
    return(predictor.item())

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
