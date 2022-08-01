# this file is for attack cases that inject kicking balls (activity M) data into walking (activity A) data

import SiameseNetWISDM
import random
import numpy as np
import torch
import copy
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if (torch.cuda.is_available()):
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# randomly sample some 2s pairs from 10s input
# in order to train the model faster and better
def randomSampleMI(X, num_pairs, length=40):  # X: 200x3
    res = []
    for i in range(num_pairs):
        idx = np.random.randint(0, X.shape[0]-length)
        x1 = X[idx:idx + length]
        idx = np.random.randint(0, X.shape[0]-length)
        x2 = X[idx:idx + length]

        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        x1 = x1.view(1, 1, 3, length)
        x2 = x2.view(1, 1, 3, length)
        res.append([x1, x2])
    return res

# segment 10s in to 9 x 2s segments with 50% overlap
# combine one with all others to get 36 pairs
def SampleMI(X, overlap = 0.5, length=40):
    x_lst = []
    start = 0
    size = X.shape[0]
    res = []
    while (start + length <= size):
        x = X[start:start + length, :]
        start += int((1-overlap) * length)
        x_lst.append(x)
    for i in range(len(x_lst)):
        for j in range(i+1, len(x_lst)):
            x1 = x_lst[i]
            x2 = x_lst[j]
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            x1 = x1.view(1, 1, 3, length)
            x2 = x2.view(1, 1, 3, length)
            res.append([x1, x2])
    if(overlap == 0.5):
        assert (len(res) == 36)
    return res

def testPerformance(net, X, Y, overlap):
    net.eval()
    num_sample = X.shape[0]
    correct = 0
    FN, FP, TP, TN = 0, 0, 0, 0
    for i in range(num_sample):
        pairs = SampleMI(X[i], overlap=overlap)
        y_pred = []
        for j in range(len(pairs)):
            x1s = pairs[j][0]
            x2s = pairs[j][1]
            if (torch.cuda.is_available()):
                x1s = x1s.cuda()
                x2s = x2s.cuda()
                net = net.cuda()
            with torch.no_grad():
                y_pred.append(net(x1s, x2s).item())
        score = min(y_pred)
        if(score >0.5):
            if(Y[i] == 1):
                correct += 1
                TP += 1
            else:
                FP += 1
        else:
            if(Y[i] == 0):
                correct += 1
                TN += 1
            else:
                FN += 1
    if(TP == 0):
        precision, recall, F1 = 0, 0, 0
    else:
        precision = TP/(TP + FP)
        recall = TP/(TP+FN)
        F1 = 2*precision*recall/(precision+recall)
    acc = correct/num_sample
    return F1, precision, recall, acc

def validationProcess(X, overlap):
    num_sample = X.shape[0]
    pairs_list = []
    for i in range(num_sample):
        pairs = SampleMI(X[i], overlap)
        pairs_list.append(pairs)
    return pairs_list

def validate(net, pairs_list, Y):
    net.eval()
    num_sample = len(pairs_list)
    correct = 0
    FN, FP, TP, TN = 0, 0, 0, 0
    for i in range(num_sample):
        pairs = pairs_list[i]
        y_pred = []
        for j in range(len(pairs)):
            x1s = pairs[j][0]
            x2s = pairs[j][1]
            if (torch.cuda.is_available()):
                x1s = x1s.cuda()
                x2s = x2s.cuda()
                net = net.cuda()
            with torch.no_grad():
                y_pred.append(net(x1s, x2s).item())
        score = min(y_pred)
        if (score > 0.5):
            if (Y[i] == 1):
                correct += 1
                TP += 1
            else:
                FP += 1
        else:
            if (Y[i] == 0):
                correct += 1
                TN += 1
            else:
                FN += 1
    if (TP == 0):
        precision, recall, F1 = 0, 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
    acc = correct / num_sample
    return F1, precision, recall, acc

def take_batch(batch_size, X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0,X.shape[0]-batch_size+1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]

def MILtrain(net, Xtrain, Ytrain, Xval, Yval, num_pairs = 15, out=None, batch_size=20, overlap=0.5, epochs=250, lr=0.0005):
    loss_func = SiameseNetWISDM.bagLoss()
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.9)
    best_F1 = 0
    VALpairs_list = validationProcess(Xval, overlap)
    model_list = []
    print("Validation data processed with overlap = {}.".format(overlap))
    for epoch in range(epochs):
        if (torch.cuda.is_available()):
            net = net.cuda()
            loss_func = loss_func.cuda()
        train_loss = []
        net.train()
        for x, y in take_batch(batch_size, Xtrain, Ytrain):
            labels = torch.from_numpy(y).view(batch_size, -1)
            if (torch.cuda.is_available()):
                labels = labels.cuda()
            y_preds = torch.empty(x.shape[0], num_pairs, 1)
            opt.zero_grad()
            for i in range(x.shape[0]):
                pairs = randomSampleMI(x[i], num_pairs)
                _, _, h, w = pairs[0][0].shape
                x1s = torch.zeros(num_pairs, 1, h, w)
                x2s = torch.zeros(num_pairs, 1, h, w)
                for j in range(len(pairs)):
                    x1 = pairs[j][0]
                    x2 = pairs[j][1]
                    x1s[j, 0, :, :] = x1
                    x2s[j, 0, :, :] = x2
                    if (torch.cuda.is_available()):
                        x1s = x1s.cuda()
                        x2s = x2s.cuda()
                y_pred = net(x1s, x2s)
                y_preds[i, :, :] = y_pred
            loss = loss_func(y_preds, labels.double())
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
        scheduler.step()
        F1, precision, recall, acc = validate(net, VALpairs_list, Yval)
        if (F1 >= best_F1):  # store the model performs best on validation set
            best_state = copy.deepcopy(net.state_dict())
            best_F1 = F1
        if (epoch > 100):
            if (best_F1 <= 0.5 and best_F1 != 0):
                break
            if (best_F1 <=0.5 and epoch%50 != 1):
                break
        if ((epoch + 1) % 50 == 0):
            best_F1 = 0
            model_list.append(best_state)

        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}...".format(F1),
              "Precision and Recall on Validation Set: {:.4f}, {:.4f}".format(precision, recall))
        # best_state = net.state_dict()
    return model_list

if __name__ == '__main__':
    groups = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
    res = {'Group1': {}, 'Group2': {}, 'Group3': {}, 'Group4': {}, 'Group5': {}}

    overlap = 0.5
    ifsmooth = False # True: train and test on smoothed attack data; False: train and test on non-smoothed data

    for group in groups:
        if(ifsmooth):
            datapath = './attackdataset/' + group + '/MinA/smooth/'
            if (overlap == 0.5):
                outpath = './results/MinA/smooth/models36/'
            else:
                outpath = './results/MinA/smooth/models15/'
        else:
            datapath = './attackdataset/' + group + '/MinA/nonsmooth/'
            if (overlap == 0.5):
                outpath = './results/MinA/nonsmooth/models36/'
            else:
                outpath = './results/MinA/nonsmooth/models15/'

        AtrainPos = np.load(file=datapath + 'trainPos.npy')
        AtestPos = np.load(file=datapath + 'testPos.npy')
        AvalPos = np.load(file=datapath + 'valPos.npy')

        # load synthetic attack samples
        # trainAtt100.npy means 1s false data is injected, trainAtt200.npy means 2s false data is injected, etc.
        AtrainAttC1s = np.load(file=datapath + 'trainAtt100.npy')
        AtrainAttC2s = np.load(file=datapath + 'trainAtt200.npy')
        AtrainAttC3s = np.load(file=datapath + 'trainAtt300.npy')
        AtrainAttC4s = np.load(file=datapath + 'trainAtt400.npy')
        AtrainAttC5s = np.load(file=datapath + 'trainAtt500.npy')

        AtestAttC1s = np.load(file=datapath + 'testAtt100.npy')
        AtestAttC2s = np.load(file=datapath + 'testAtt200.npy')
        AtestAttC3s = np.load(file=datapath + 'testAtt300.npy')
        AtestAttC4s = np.load(file=datapath + 'testAtt400.npy')
        AtestAttC5s = np.load(file=datapath + 'testAtt500.npy')

        AvalAttC1s = np.load(file=datapath + 'valAtt100.npy')
        AvalAttC2s = np.load(file=datapath + 'valAtt200.npy')
        AvalAttC3s = np.load(file=datapath + 'valAtt300.npy')
        AvalAttC4s = np.load(file=datapath + 'valAtt400.npy')
        AvalAttC5s = np.load(file=datapath + 'valAtt500.npy')

        Xtrain = np.vstack([AtrainPos, AtrainAttC1s, AtrainAttC2s, AtrainAttC3s, AtrainAttC4s, AtrainAttC5s])
        YtrainPure = np.ones(AtrainPos.shape[0])
        YtrainSyn = np.zeros(AtrainAttC1s.shape[0] * 5)
        Ytrain = np.hstack([YtrainPure, YtrainSyn])

        Xtest = np.vstack([AtestPos, AtestAttC1s, AtestAttC2s, AtestAttC3s, AtestAttC4s, AtestAttC5s])
        YtestPure = np.ones(AtestPos.shape[0])
        YtestSyn = np.zeros(AtestAttC1s.shape[0] * 5)
        Ytest = np.hstack([YtestPure, YtestSyn])

        Xval = np.vstack([AvalPos, AvalAttC1s, AvalAttC2s, AvalAttC3s, AvalAttC4s, AvalAttC5s])
        YvalPure = np.ones(AvalPos.shape[0])
        YvalSyn = np.zeros(AvalAttC1s.shape[0] * 5)
        Yval = np.hstack([YvalPure, YvalSyn])
        print("Data Loaded.")

        print("Number of training samples : {}".format(Xtrain.shape[0]))
        print("Number of testing samples : {}".format(Xtest.shape[0]))
        print("Number of validation samples : {}".format(Xval.shape[0]))

        net = SiameseNetWISDM.SiameseNetwork2s()
        model_list = MILtrain(net, Xtrain, Ytrain, Xval, Yval, overlap=overlap, epochs=250)

        best_model = SiameseNetWISDM.SiameseNetwork2s()
        count = 1
        res[group] = {"F1": [], "precision": [], "recall": [],
                      "1s": [], "2s": [], "3s": [], "4s": [], "5s": []}
        for best_state in model_list:
            torch.save(best_state, outpath + group + '_' + str(count) + '_MILmodel.pth')
            count += 1
            best_model.load_state_dict(best_state)

            F1, precision, recall, acc = testPerformance(best_model, Xtest, Ytest, overlap=overlap)
            res[group]['F1'].append(F1)
            res[group]['precision'].append(precision)
            res[group]['recall'].append(recall)

            synY = np.zeros((AtestAttC1s.shape[0], 1))

            acc1s = testPerformance(best_model, AtestAttC1s, synY, overlap=overlap)[3]
            acc2s = testPerformance(best_model, AtestAttC2s, synY, overlap=overlap)[3]
            acc3s = testPerformance(best_model, AtestAttC3s, synY, overlap=overlap)[3]
            acc4s = testPerformance(best_model, AtestAttC4s, synY, overlap=overlap)[3]
            acc5s = testPerformance(best_model, AtestAttC5s, synY, overlap=overlap)[3]

            res[group]["1s"].append(acc1s)
            res[group]["2s"].append(acc2s)
            res[group]["3s"].append(acc3s)
            res[group]["4s"].append(acc4s)
            res[group]["5s"].append(acc5s)


            f = open(outpath + '2s36.txt', 'w')
            f.write(str(res))
            f.close()

            print("In {}:".format(group))
            print("F1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(F1, precision, recall))
            print("acc 1s: {:.4f}, acc 2s: {:.4f}, acc 3s: {:.4f}, acc 4s: {:.4f}, acc 5s: {:.4f}".format(acc1s, acc2s,
                                                                                                       acc3s, acc4s,
                                                                                                       acc5s))
