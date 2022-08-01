import torch
from torch import nn

class bagLoss(nn.Module):
    def __init__(self):
        super(bagLoss, self).__init__()

    def forward(self, y_pred, label):
        if (torch.cuda.is_available()):
            y_pred = y_pred.cuda()
            label = label.cuda()
        lossfunc = torch.nn.BCELoss()
        loss = lossfunc(torch.min(y_pred, 1)[0].double(), label.double())
        return loss

def L1_distance(y1, y2):
    num_sample = y1.size()[0]
    out = torch.zeros((num_sample, 1))
    for i in range(num_sample):
        dist = torch.dist(y1[i], y2[i], 1)
        out[i] = dist
    return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class SiameseNetwork4s(nn.Module):
    def __init__(self):
        super(SiameseNetwork4s, self).__init__()
        self.cnn1 = nn.Sequential(nn.Conv1d(1, 256, 3, stride=2),
                                  nn.LeakyReLU(inplace=True),
                                  nn.MaxPool1d(2, stride=2))
        self.cnn2 = nn.Sequential(nn.Conv1d(256, 128, 4, stride=3),
                                  nn.LeakyReLU(inplace=True),
                                  nn.MaxPool1d(3, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(1152, 512),
                                 nn.Dropout(0.3),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(512, 256),
                                 nn.Dropout(0.3),
                                 #nn.LeakyReLU(inplace=True),
                                 #nn.Linear(256, 150),
                                 #nn.Dropout(0.3),
                                 nn.LeakyReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(1, 1),
                                 nn.Sigmoid())

    def forward_once(self, x):
        output = self.cnn1(x)
        output = self.cnn2(output)
        output = output.view(output.size(0), -1)
        y = self.fc1(output)
        return y

    def forward(self, x1, x2):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        dist = L1_distance(y1, y2)
        if (torch.cuda.is_available()):
            dist = dist.cuda()
        f = self.fc2(dist)
        return f