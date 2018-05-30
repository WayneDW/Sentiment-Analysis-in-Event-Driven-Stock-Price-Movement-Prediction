import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def train(X_train, y_train, X_valid, y_valid, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, last_step = 0, 0
    model.train()
    batch = args.batch_size
    for epoch in range(1, args.epochs+1):
        for idx in range(int(X_train.shape[0]/batch)):
            feature = torch.LongTensor(X_train[(idx*batch):(idx*batch+batch),])
            target = torch.LongTensor(y_train[(idx*batch):(idx*batch+batch)])
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            '''
            # my new part
            myforward = F.log_softmax(feature, dim=1)
            nllloss = nn.NLLLoss(size_average = False)
            loss = nllloss.forward(myforward, target)
            '''
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum().item()
                accuracy = 100.0 * corrects / batch
                sys.stdout.write('\rEpoch[{}] Batch[{}] - loss: {:.4f}  acc: {:.2f}%({}/{})'.format(
                                 epoch, idx, loss.item(), accuracy, corrects, batch))
        dev_acc = eval(X_valid, y_valid, model, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            last_step = idx
            if args.save_best:
                save(model, args.save_dir, 'best', epoch)
        save(model, args.save_dir, 'snapshot', epoch)


def eval(X, y, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    correct_part, total_part = 0, 0
    batch = args.batch_size
    for idx in range(int(X.shape[0]/batch)):
        feature = torch.LongTensor(X[(idx*batch):(idx*batch+batch),])
        target = torch.LongTensor(y[(idx*batch):(idx*batch+batch)])
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        '''
        # my new part
        logit = F.log_softmax(feature, dim=1)
        nllloss = nn.NLLLoss(size_average = False)
        loss = nllloss.forward(logit, target)
        '''
        loss = F.cross_entropy(logit, target, size_average=False)
        #print(logit)
        avg_loss += loss.data.item()
        predictor = torch.exp(logit[:, 1]) / (torch.exp(logit[:, 0]) + torch.exp(logit[:, 1]))
        thres = 0.1
        idx_thres = (predictor > 0.5 + thres) + (predictor < 0.5 - thres)
        #print(predictor)
        #print(torch.max(logit, 1)[1])
        #print(if_condition)
        #print(torch.max(logit, 1)[1].view(target.size()).data)

        corrects += (torch.max(logit, 1)[1] == target.data).sum().item()
        correct_part += (torch.max(logit, 1)[1][idx_thres] == target.data[idx_thres]).sum().item()
        total_part += idx_thres.sum().item()

        
        #corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum().item()

    size = y.shape[0]
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\n')
    print(100.0 * correct_part / total_part, correct_part, total_part)
    print('\n         Evaluation - loss: {:.4f}  acc: {:.2f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
