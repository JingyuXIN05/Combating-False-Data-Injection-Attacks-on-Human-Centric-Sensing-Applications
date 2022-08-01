import numpy as np
import pandas as pd
import os
import random

seed = 1
random.seed(seed)
np.random.seed(seed)

def remove_col(df):
    return df.drop(['timestamp'], axis=1)

def sample(path, file, length):
    df = pd.read_csv(path+file)
    df = remove_col(df)
    idx = np.random.randint(0, len(df))
    while(idx + length >= len(df)):
        idx = np.random.randint(0, len(df))
    res = df.iloc[idx:idx+length].values
    return res

def ExpMovingAverage(array, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(array, weights, mode='full')[:len(array)]
    a[:window] = a[window]
    return a

# window = 4 means 0.2s
def smooth(copy, pos, window = 4):
    res = copy.copy()
    start = pos - 10
    end = pos + 10
    data_x = copy[start:end,0]
    data_y = copy[start:end,1]
    data_z = copy[start:end,2]
    smoothed_x = ExpMovingAverage(data_x, window)
    smoothed_y = ExpMovingAverage(data_y, window)
    smoothed_z = ExpMovingAverage(data_z, window)
    res[start + window +1 : end - window, 0] = smoothed_x[window+1:-window]
    res[start + window +1 : end - window, 1] = smoothed_y[window+1:-window]
    res[start + window +1 : end - window, 2] = smoothed_z[window+1:-window]
    return res

def forgeSample(originalPath, usrGroup, corruptionLen, corruptionPath, ifSmooth, num, puresize = 200):
    attackSamples = []
    corruptionFiles = os.listdir(corruptionPath)
    for file in os.listdir(originalPath):
        if(file not in usrGroup):
            continue
        df = pd.read_csv(originalPath + file)
        if (len(df) < puresize):
            continue
        for i in range(num):
            extractFrom = np.random.randint(0, len(corruptionFiles))
            while (len(pd.read_csv(corruptionPath + corruptionFiles[extractFrom])) < corruptionLen
                   or corruptionFiles[extractFrom] == file
                   or corruptionFiles[extractFrom] not in usrGroup):
                extractFrom = np.random.randint(0, len(corruptionFiles))
            corruption = sample(corruptionPath, corruptionFiles[extractFrom], corruptionLen)
            pureSample = sample(originalPath, file, puresize)

            start = np.random.randint(20, puresize - corruptionLen - 20)
            end = start + corruptionLen
            pureSample[start:end, :] = corruption

            if (ifSmooth):
                pureSample = smooth(pureSample, start)
                pureSample = smooth(pureSample, end)
            pureSample = pureSample[np.newaxis, :]
            attackSamples.append(pureSample)
    attackSamples = np.vstack(attackSamples)
    print("Number of attack {} samples: {}".format(corruptionLen, len(attackSamples)))
    return attackSamples

def postiveSamples(path, usrGroup, num, pureSize = 200):
    posSamples = []
    for file in os.listdir(path):
        if(file not in usrGroup):
            continue
        df = pd.read_csv(path + file)
        if(len(df) < pureSize):
            continue
        for i in range(num):
            pureSample = sample(path, file, pureSize)
            pureSample = pureSample[np.newaxis, :]
            posSamples.append(pureSample)
    print("Number of pos samples: {}".format(len(posSamples)))
    posSamples = np.vstack(posSamples)
    return posSamples

def SampleGenerator(originalpath, attackpath, numEach, ifSmooth, outpath, trainGroup, testGroup, valGroup):
    AtrainPos = postiveSamples(originalpath, trainGroup, numEach)
    AtestPos = postiveSamples(originalpath, testGroup, numEach)
    AvalPos = postiveSamples(originalpath, valGroup, numEach)
    np.save(file=outpath + 'trainPos.npy', arr=AtrainPos)
    np.save(file=outpath + 'testPos.npy', arr=AtestPos)
    np.save(file=outpath + 'valPos.npy', arr=AvalPos)

    AtrainAttB1s = forgeSample(originalpath, trainGroup, 20, attackpath, ifSmooth, int(numEach / 5))
    AtrainAttB2s = forgeSample(originalpath, trainGroup, 40, attackpath, ifSmooth, int(numEach / 5))
    AtrainAttB3s = forgeSample(originalpath, trainGroup, 60, attackpath, ifSmooth, int(numEach / 5))
    AtrainAttB4s = forgeSample(originalpath, trainGroup, 80, attackpath, ifSmooth, int(numEach / 5))
    AtrainAttB5s = forgeSample(originalpath, trainGroup, 100, attackpath, ifSmooth, int(numEach / 5))

    AtestAttB1s = forgeSample(originalpath, testGroup, 20, attackpath, ifSmooth, int(numEach / 5))
    AtestAttB2s = forgeSample(originalpath, testGroup, 40, attackpath, ifSmooth, int(numEach / 5))
    AtestAttB3s = forgeSample(originalpath, testGroup, 60, attackpath, ifSmooth, int(numEach / 5))
    AtestAttB4s = forgeSample(originalpath, testGroup, 80, attackpath, ifSmooth, int(numEach / 5))
    AtestAttB5s = forgeSample(originalpath, testGroup, 100, attackpath, ifSmooth, int(numEach / 5))

    AvalAttB1s = forgeSample(originalpath, valGroup, 20, attackpath, ifSmooth, int(numEach / 5))
    AvalAttB2s = forgeSample(originalpath, valGroup, 40, attackpath, ifSmooth, int(numEach / 5))
    AvalAttB3s = forgeSample(originalpath, valGroup, 60, attackpath, ifSmooth, int(numEach / 5))
    AvalAttB4s = forgeSample(originalpath, valGroup, 80, attackpath, ifSmooth, int(numEach / 5))
    AvalAttB5s = forgeSample(originalpath, valGroup, 100, attackpath, ifSmooth, int(numEach / 5))

    np.save(file=outpath + 'trainAtt100.npy', arr=AtrainAttB1s)
    np.save(file=outpath + 'trainAtt200.npy', arr=AtrainAttB2s)
    np.save(file=outpath + 'trainAtt300.npy', arr=AtrainAttB3s)
    np.save(file=outpath + 'trainAtt400.npy', arr=AtrainAttB4s)
    np.save(file=outpath + 'trainAtt500.npy', arr=AtrainAttB5s)

    np.save(file=outpath + 'testAtt100.npy', arr=AtestAttB1s)
    np.save(file=outpath + 'testAtt200.npy', arr=AtestAttB2s)
    np.save(file=outpath + 'testAtt300.npy', arr=AtestAttB3s)
    np.save(file=outpath + 'testAtt400.npy', arr=AtestAttB4s)
    np.save(file=outpath + 'testAtt500.npy', arr=AtestAttB5s)

    np.save(file=outpath + 'valAtt100.npy', arr=AvalAttB1s)
    np.save(file=outpath + 'valAtt200.npy', arr=AvalAttB2s)
    np.save(file=outpath + 'valAtt300.npy', arr=AvalAttB3s)
    np.save(file=outpath + 'valAtt400.npy', arr=AvalAttB4s)
    np.save(file=outpath + 'valAtt500.npy', arr=AvalAttB5s)

if __name__ == '__main__':
    Apath = "./OriginalData/actA/"
    Bpath = "./OriginalData/actB/"
    Cpath = "./OriginalData/actC/"
    Mpath = "./OriginalData/actM/"

    f = open('./GroupSplit.txt', 'r')
    usrSplit = f.read()
    usrSplit = eval(usrSplit)
    f.close()

    ifSmooth = False
    numEach = 40
    for group in usrSplit.keys():
        testGroup = usrSplit[group]['testing']
        trainGroup = usrSplit[group]['training']
        valGroup = usrSplit[group]['validation']
        # attack A using B
        if (ifSmooth):
            outpath = './attackdataset/'+ group + '/BinA/smooth/'
        else:
            outpath = './attackdataset/'+ group + '/BinA/nonsmooth/'

        SampleGenerator(Apath, Bpath, numEach, ifSmooth, outpath, trainGroup, testGroup, valGroup)

        # attack A using C
        if (ifSmooth):
            outpath = './attackdataset/' + group + '/CinA/smooth/'
        else:
            outpath = './attackdataset/' + group + '/CinA/nonsmooth/'

        SampleGenerator(Apath, Cpath, numEach, ifSmooth, outpath, trainGroup, testGroup, valGroup)

        # attack A using M
        if (ifSmooth):
            outpath = './attackdataset/' + group + '/MinA/smooth/'
        else:
            outpath = './attackdataset/' + group + '/MinA/nonsmooth/'

        SampleGenerator(Apath, Mpath, numEach, ifSmooth, outpath, trainGroup, testGroup, valGroup)

        # attack C using A
        if (ifSmooth):
            outpath = './attackdataset/' + group + '/AinC/smooth/'
        else:
            outpath = './attackdataset/' + group + '/AinC/nonsmooth/'

        SampleGenerator(Cpath, Apath, numEach, ifSmooth, outpath, trainGroup, testGroup, valGroup)

        # attack C using B
        if (ifSmooth):
            outpath = './attackdataset/' + group + '/BinC/smooth/'
        else:
            outpath = './attackdataset/' + group + '/BinC/nonsmooth/'

        SampleGenerator(Cpath, Bpath, numEach, ifSmooth, outpath, trainGroup, testGroup, valGroup)

