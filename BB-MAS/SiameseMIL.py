import SiameseNetBBMAS
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
def randomSampleMI(X, num_pairs, length=200):  # X: 1000x3
    res = []
    for i in range(num_pairs):
        idx = np.random.randint(0, X.shape[0] - length)
        x1 = X[idx:idx + length]
        idx = np.random.randint(0, X.shape[0] - length)
        x2 = X[idx:idx + length]
        x1 = x1.T
        x2 = x2.T
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        x1 = x1.view(1, 1, 3, length)
        x2 = x2.view(1, 1, 3, length)
        res.append([x1, x2])
    return res


# segment 10s in to 9 x 2s segments with 50% overlap
# combine one with all others to get 36 pairs
def SampleMI(X, overlap=0.5, length=200):
    x_lst = []
    start = 0
    size = X.shape[0]
    res = []
    while (start + length <= size):
        x = X[start:start + length, :]
        start += int((1 - overlap) * length)
        x_lst.append(x)
    for i in range(len(x_lst)):
        for j in range(i + 1, len(x_lst)):
            x1 = x_lst[i]
            x2 = x_lst[j]
            x1 = x1.T
            x2 = x2.T
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            x1 = x1.view(1, 1, 3, length)
            x2 = x2.view(1, 1, 3, length)
            res.append([x1, x2])
    if (overlap == 0.5):
        assert (len(res) == 36)
    return res


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
        # print(score)
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
    for i in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]


def MILtrain(net, Xtrain, Ytrain, Xval, Yval, num_pairs=15, batch_size=40, overlap=0.5, epochs=160, lr=0.0005):
    loss_func = SiameseNetBBMAS.bagLoss()
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
        if ((epoch + 1) % 50 == 0): # save the best model in every 50 epochs
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
    ifsmooth = True # set to false if train and test on non-smoothed data
    for group in groups:
        if (ifsmooth):
            datapath = './Forge-replayData/' + group + '/smooth/'
            if (overlap == 0.5):
                outpath = './newres/smooth/models36/'
            else:
                outpath = './newres/smooth/models15/'
        else:
            datapath = './Forge-replayData/' + group + '/nonsmooth/'
            if (overlap == 0.5):
                outpath = './newres/nonsmooth/models36/'
            else:
                outpath = './newres/nonsmooth/models15/'

        Xtrain = np.load(file=datapath + 'trainX.npy')
        Ytrain = np.load(file=datapath + 'trainY.npy')

        Xtest = np.load(file=datapath + 'testX.npy')
        Ytest = np.load(file=datapath + 'testY.npy')

        Xval = np.load(file=datapath + 'valX.npy')
        Yval = np.load(file=datapath + 'valY.npy')
        print("Start training...")
        net = SiameseNetBBMAS.SiameseNetwork2s()
        # net.apply(SiameseNetBBMAS.init_weights)
        model_list = MILtrain(net, Xtrain, Ytrain, Xval, Yval, overlap=overlap, epochs=250)

        print("Start evaluation...")
        best_model = SiameseNetBBMAS.SiameseNetwork2s()
        count = 1
        res[group] = {"F1": [], "precision": [], "recall": []}
        # loop over the best models in every 50 epochs
        # record their performance on test set
        for best_state in model_list:
            torch.save(best_state, outpath + group + '_' + str(count) + '_MILmodel.pth')
            count += 1
            best_model.load_state_dict(best_state)

            F1, precision, recall, acc = testPerformance(best_model, Xtest, Ytest, overlap=overlap)
            res[group]['F1'].append(F1)
            res[group]['precision'].append(precision)
            res[group]['recall'].append(recall)

            f = open(outpath + '2s36.txt', 'w')
            f.write(str(res))
            f.close()

            print("In {}:".format(group))
            print("F1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(F1, precision, recall))


