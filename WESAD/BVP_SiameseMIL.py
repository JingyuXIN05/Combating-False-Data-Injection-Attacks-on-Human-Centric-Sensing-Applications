import SiameseNetWESAD
import torch
import numpy as np
import pickle
import copy

# segment 20s in to 9 x 4s segments with 50% overlap
# combine one with all others to get 36 pairs
def SampleMI4s(X, overlap=0.5, length=256):  # 4s = 256
    x_lst = []
    start = 0
    size = X.shape[1]
    res = []
    while (start + length <= size):
        x = X[:, start:start + length]
        start += int((1 - overlap) * length)
        x_lst.append(x)
    for i in range(len(x_lst)):
        for j in range(i + 1, len(x_lst)):
            x1 = x_lst[i]
            x2 = x_lst[j]
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            x1 = x1.view(1, 1, length)  # change it into (1, 1, 1,length) when using Conv2D
            x2 = x2.view(1, 1, length)
            res.append([x1, x2])
    # print(len(x_lst), len(res))
    if (overlap == 0.25):
        assert (len(res) == 15)
    if (overlap == 0.5):
        assert (len(res) == 36)
    return res

# randomly sample some 4s pairs from 20s input
# in order to train the model faster and better
def randomSampleMI4s(X, num_pairs, length=256):  # only used in training
    res = []
    for i in range(num_pairs):
        idx = np.random.randint(0, X.shape[1] - length)
        x1 = X[:, idx:idx + length]
        idx = np.random.randint(0, X.shape[1] - length)
        x2 = X[:, idx:idx + length]
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        x1 = x1.view(1, 1, length)
        x2 = x2.view(1, 1, length)
        res.append([x1, x2])
    return res


def validationProcess(X, overlap):
    num_sample = X.shape[0]
    pairs_list = []
    for i in range(num_sample):
        pairs = SampleMI4s(X[i], overlap=overlap)
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
    for i in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[i:i + batch_size]
        yield X[excerpt], Y[excerpt]


def testPerformance(net, X, Y, overlap):
    net.eval()
    num_sample = X.shape[0]
    correct = 0
    FN, FP, TP, TN = 0, 0, 0, 0
    for i in range(num_sample):
        pairs = SampleMI4s(X[i], overlap=overlap)
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


def MILtrain4s(net, Xtrain, Ytrain, Xval, Yval, num_pairs=15, batch_size=32, overlap=0.5, epochs=300, lr=0.0005):
    loss_func = SiameseNetWESAD.bagLoss()
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.9)
    best_F1 = 0
    VALpairs_list = validationProcess(Xval, overlap)
    model_list = []
    print("Validation data processed.")
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
            y_preds = torch.empty(batch_size, num_pairs, 1)
            opt.zero_grad()
            for i in range(batch_size):
                # pairs = SampleMI4s(x[i], overlap=overlap)
                pairs = randomSampleMI4s(x[i], num_pairs)
                _, _, w = pairs[0][0].shape
                x1s = torch.empty(num_pairs, 1, w)
                x2s = torch.empty(num_pairs, 1, w)
                for j in range(len(pairs)):
                    x1 = pairs[j][0]
                    x2 = pairs[j][1]
                    x1s[j, 0, :] = x1
                    x2s[j, 0, :] = x2
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
        if ((epoch + 1) % 50 == 0):
            best_F1 = 0
            model_list.append(best_state)
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_loss)),
              "Validation F1: {:.4f}...".format(F1),
              "Precision and Recall on Validation Set: {:.4f}, {:.4f}".format(precision, recall))
    return model_list


if __name__ == '__main__':
    groups = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
    res = {'Group1': {}, 'Group2': {}, 'Group3': {}, 'Group4': {}, 'Group5': {}}

    ifsmooth = True # False: train and test on non-smoothed data; True: train and test on smoothed data
    overlap = 0.5
    print("ifsmooth = {}, overlap = {}".format(ifsmooth, overlap))
    for group in groups:
        path = '/home/jxin05/WESAD_BVP/bvp_data/' + group + '/'

        print("Loading training data...")
        if (ifsmooth):
            negTrainPath = path + 'train/PosinNeg/smooth/'
        else:
            negTrainPath = path + 'train/PosinNeg/nonsmooth/'

        with open(path + 'train/pos/windows.pkl', 'rb') as fp:
            postrain1 = pickle.load(fp)
        with open(path + 'train/neg/windows.pkl', 'rb') as fp:
            postrain2 = pickle.load(fp)
        posTrainSamples = []
        for key in postrain1.keys():
            for window in postrain1[key]:
                x = window.values[:, 0]
                x = x.reshape(1, -1)
                posTrainSamples.append(x)
            for window in postrain2[key]:
                x = window.values[:, 0]
                x = x.reshape(1, -1)
                posTrainSamples.append(x)
        posTrainSamples = np.array(posTrainSamples)
        print(posTrainSamples.shape)

        train2s = np.load(file=negTrainPath + "trainX2s.npy")
        train4s = np.load(file=negTrainPath + "trainX4s.npy")
        train6s = np.load(file=negTrainPath + "trainX6s.npy")
        train8s = np.load(file=negTrainPath + "trainX8s.npy")
        train10s = np.load(file=negTrainPath + "trainX10s.npy")
        negTrainSamples = np.vstack([train2s, train4s, train6s, train8s, train10s])
        num = negTrainSamples.shape[0]
        length = negTrainSamples.shape[1]
        negTrainSamples = negTrainSamples.reshape(num, 1, length)
        print(negTrainSamples.shape)

        print("Loading testing data...")
        if (ifsmooth):
            negTestPath = path + 'test/PosinNeg/smooth/'
        else:
            negTestPath = path + 'test/PosinNeg/nonsmooth/'

        with open(path + 'test/pos/windows.pkl', 'rb') as fp:
            postest1 = pickle.load(fp)
        with open(path + 'test/neg/windows.pkl', 'rb') as fp:
            postest2 = pickle.load(fp)

        posTestSamples = []
        for key in postest1.keys():
            for window in postest1[key]:
                x = window.values[:, 0]
                x = x.reshape(1, -1)
                posTestSamples.append(x)
            for window in postest2[key]:
                x = window.values[:, 0]
                x = x.reshape(1, -1)
                posTestSamples.append(x)
        posTestSamples = np.array(posTestSamples)
        print(posTestSamples.shape)

        test2s = np.load(file=negTestPath + "testX2s.npy")
        test4s = np.load(file=negTestPath + "testX4s.npy")
        test6s = np.load(file=negTestPath + "testX6s.npy")
        test8s = np.load(file=negTestPath + "testX8s.npy")
        test10s = np.load(file=negTestPath + "testX10s.npy")

        n, h, w = test2s.shape

        negTestSamples = np.vstack([test2s, test4s, test6s, test8s, test10s])
        num = negTestSamples.shape[0]
        length = negTestSamples.shape[1]
        negTestSamples = negTestSamples.reshape(num, 1, length)

        test2s = test2s.reshape(n, 1, h)
        test4s = test4s.reshape(n, 1, h)
        test6s = test6s.reshape(n, 1, h)
        test8s = test8s.reshape(n, 1, h)
        test10s = test10s.reshape(n, 1, h)

        print(negTestSamples.shape)

        print("Loading validation data...")
        if (ifsmooth):
            negValPath = path + 'validation/PosinNeg/smooth/'
        else:
            negValPath = path + 'validation/PosinNeg/nonsmooth/'

        with open(path + 'validation/pos/windows.pkl', 'rb') as fp:
            posval1 = pickle.load(fp)
        with open(path + 'validation/neg/windows.pkl', 'rb') as fp:
            posval2 = pickle.load(fp)

        posValSamples = []
        for key in posval1.keys():
            for window in posval1[key]:
                x = window.values[:, 0]
                x = x.reshape(1, -1)
                posValSamples.append(x)
            for window in posval2[key]:
                x = window.values[:, 0]
                x = x.reshape(1, -1)
                posValSamples.append(x)
        posValSamples = np.array(posValSamples)
        print(posValSamples.shape)

        val2s = np.load(file=negValPath + "valX2s.npy")
        val4s = np.load(file=negValPath + "valX4s.npy")
        val6s = np.load(file=negValPath + "valX6s.npy")
        val8s = np.load(file=negValPath + "valX8s.npy")
        val10s = np.load(file=negValPath + "valX10s.npy")
        negValSamples = np.vstack([val2s, val4s, val6s, val8s, val10s])
        num = negValSamples.shape[0]
        length = negValSamples.shape[1]
        negValSamples = negValSamples.reshape(num, 1, length)
        print(negValSamples.shape)
        print("Data loading finished.")

        trainX = np.vstack([posTrainSamples, negTrainSamples])
        trainY = np.hstack([np.ones(posTrainSamples.shape[0]), np.zeros(negTrainSamples.shape[0])])

        valX = np.vstack([posValSamples, negValSamples])
        valY = np.hstack([np.ones(posValSamples.shape[0]), np.zeros(negValSamples.shape[0])])

        testX = np.vstack([posTestSamples, negTestSamples])
        testY = np.hstack([np.ones(posTestSamples.shape[0]), np.zeros(negTestSamples.shape[0])])

        net = SiameseNetWESAD.SiameseNetwork4s()
        # net.apply(init_weights)
        if (ifsmooth):
            out = '/home/jxin05/WESAD_BVP/res/smooth/models/'
        else:
            out = '/home/jxin05/WESAD_BVP/res/nonsmooth/models/'

        model_list = MILtrain4s(net, trainX, trainY, valX, valY, overlap=overlap, epochs=250)

        best_model = SiameseNetWESAD.SiameseNetwork4s()
        print("Start evaluation...")
        count = 1
        res[group] = {"F1": [], "precision": [], "recall": [],
                      "2s": [], "4s": [], "6s": [], "8s": [], "10s": []}
        for best_state in model_list:
            torch.save(best_state, out + group + '_' + str(count) + '_BVP_MILmodel4s.pth')
            count += 1
            best_model.load_state_dict(best_state)

            # modeal evaluation

            F1, precision, recall, acc = testPerformance(best_model, testX, testY, overlap=overlap)
            res[group]["F1"].append(F1)
            res[group]["precision"].append(precision)
            res[group]["recall"].append(recall)

            synY = np.zeros((test2s.shape[0], 1))
            acc2s = testPerformance(best_model, test2s, synY, overlap=overlap)[3]
            acc4s = testPerformance(best_model, test4s, synY, overlap=overlap)[3]
            acc6s = testPerformance(best_model, test6s, synY, overlap=overlap)[3]
            acc8s = testPerformance(best_model, test8s, synY, overlap=overlap)[3]
            acc10s = testPerformance(best_model, test10s, synY, overlap=overlap)[3]

            res[group]["2s"].append(acc2s)
            res[group]["4s"].append(acc4s)
            res[group]["6s"].append(acc6s)
            res[group]["8s"].append(acc8s)
            res[group]["10s"].append(acc10s)

            if (overlap == 0.5):
                f = open(out + 'BVP_4s36last.txt', 'w')
            else:
                f = open(out + 'BVP_4s15last.txt', 'w')
            f.write(str(res))
            f.close()

            print("In {}:".format(group))
            print("F1: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(F1, precision, recall))
            print("acc 2s: {:.4f}, acc 4s: {:.4f}, acc 6s: {:.4f}, acc 8s: {:.4f}, acc 10s: {:.4f}".format(acc2s, acc4s,
                                                                                                           acc6s, acc8s,
                                                                                                           acc10s))