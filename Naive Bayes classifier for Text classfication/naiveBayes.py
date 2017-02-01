#!/usr/bin/python

import sys
import os
import numpy as np
import glob as gb
import re
import math
from sklearn.naive_bayes import MultinomialNB

###############################################################################


vocabulary = ["love", "wonderful", "best", "great", "superb", "still", "beautiful",
              "bad", "worst", "stupid", "waste", "boring", "\?", "\!", "UNK"]



def transfer(fileDj, vocabulary):
    BOWDj = [0 for i in range(15)]

    f = open(fileDj, "r")
    content = f.readlines()

    for i in range(len(content)):
        content[i] = content[i].replace("loves", "love")
        content[i] = content[i].replace("loved", "love")
        content[i] = content[i].replace("loving", "love")
        for j in range(14):
            BOWDj[j] += len(re.findall(vocabulary[j], content[i]) )
            #BOWDj[14] -= BOWDj[j]
        BOWDj[14] += len(content[i].split())
    return BOWDj


def loadData(Path):

    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []

    #### train_set
    str = Path + "/training_set" + "/pos" +"/*.txt"
    names = gb.glob(str)
    for filename in names:
        BOWDj = transfer(filename, vocabulary)
        Xtrain.append(BOWDj)
        ytrain.append(1)
    #print "Trainging_set pos done"

    str = Path + "/training_set" + "/neg" + "/*.txt"
    names = gb.glob(str)
    for filename in names:
        BOWDj = transfer(filename, vocabulary)
        Xtrain.append(BOWDj)
        ytrain.append(0)
    #print "Trainging_set neg done"


    #### test_set
    str = Path + "/test_set" + "/pos" + "/*.txt"
    names = gb.glob(str)
    for filename in names:
        BOWDj = transfer(filename, vocabulary)
        Xtest.append(BOWDj)
        ytest.append(1)
    #print "test_set pos done"

    str = Path + "/test_set" + "/neg" + "/*.txt"
    names = gb.glob(str)
    for filename in names:
        BOWDj = transfer(filename, vocabulary)
        Xtest.append(BOWDj)
        ytest.append(0)

    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    thetaPos = [0.0 for i in range(15)]
    thetaNeg = [0.0 for i in range(15)]
    numNeg = 0
    numPos = 0

    for i in range(len(Xtrain)):    # i samples
        if ytrain[i] == 0:              # Negative sample
            for j in range(15):         # j words
                thetaNeg[j] += Xtrain[i][j]

        else:                           # Postive sample
            for j in range(15):         # j words
                thetaPos[j] += Xtrain[i][j]

    for i in range(15):
        numNeg += thetaNeg[i]
        numPos += thetaPos[i]

    for i in range(15): # MLE estimator plus smoothing
        thetaPos[i] = (thetaPos[i] + 1)*1.0 / (numPos + 15)
        thetaNeg[i] = (thetaNeg[i] + 1)*1.0 / (numNeg + 15)

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    Accuracy = 0.0
    pos = 0
    neg = 0
    yPredict = []
    correct = 0

    for i in range(0, len(Xtest)):
        pos = neg = 0
        for j in range(0, 15):
            pos += Xtest[i][j] * math.log(thetaPos[j])
            neg += Xtest[i][j] * math.log(thetaNeg[j])
        if pos > neg:
            yPredict.append(1)
        else:
            yPredict.append(0)
        if yPredict[i] == ytest[i]:
            correct += 1
    Accuracy = correct*1.0/len(Xtest)
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):

    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    correct_skl = 0
    for i in range(0, len(Xtest)):
        if ytest[i] == clf.predict(Xtest[i:i+1]):
            correct_skl += 1
    Accuracy = correct_skl*1.0/len(Xtest)
    return Accuracy



def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg, vocabulary):
    f = open(path, "r")
    content = f.readlines()
    pos = 0.0
    neg = 0.0

    for i in range(len(content)):
        content[i] = content[i].replace("loves", "love")
        content[i] = content[i].replace("loved", "love")
        content[i] = content[i].replace("loving", "love")
        for j in range(14):
            pos += len(re.findall(vocabulary[j], content[i])) * math.log(thetaPos[j])
            neg += len(re.findall(vocabulary[j], content[i])) * math.log(thetaNeg[j])
        pos += math.log(thetaPos[14]) * len(content[i].split())
        neg += math.log(thetaNeg[14]) * len(content[i].split())

    if pos > neg: yPredict = 1
    else: yPredict = 0
    return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg, vocabulary):
    yPredict = []
    correct = 0
    num = 0
    ################# POS
    str = path + "/pos" + "/*.txt"
    names = gb.glob(str)
    for filename in names:
        predict = naiveBayesMulFeature_testDirectOne(filename,thetaPos, thetaNeg, vocabulary)
        yPredict.append(predict)
        num += 1
        if predict == 1:
            correct += 1

    ################### NEG
    str = path + "/neg" + "/*.txt"
    names = gb.glob(str)
    for filename in names:
        predict = naiveBayesMulFeature_testDirectOne(filename, thetaPos, thetaNeg, vocabulary)
        yPredict.append(predict)
        num += 1
        if predict == 0:
            correct += 1

    Accuracy = correct*1.0 / num

    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
    pos = 0
    neg = 0
    thetaPosTrue = [0.0 for i in range(15)]
    thetaNegTrue = [0.0 for i in range(15)]

    for i in range(len(Xtrain)):
        for j in range(15):
            if Xtrain[i][j] != 0:          # contain the specific word
                if ytrain[i] == 1:      # this sample is pos
                    thetaPosTrue[j] += 1
                else:
                    thetaNegTrue[j] += 1
        if ytrain[i] == 1:
            pos += 1
        else:
            neg += 1

    for i in range(15):
        thetaPosTrue[i] = thetaPosTrue[i]*1.0/pos
        thetaNegTrue[i] = thetaNegTrue[i]*1.0/neg

    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    Accuracy = 0.0
    correct = 0
    pos = 0.0
    neg = 0.0

    for i in range(len(Xtest)):
        pos = 0.0
        neg = 0.0
        for j in range(15):
            if(Xtest[i][j] != 0):
                pos += math.log(thetaPosTrue[j])
                neg += math.log(thetaNegTrue[j])
        if pos > neg:
            yPredict.append(1)
        else:
            yPredict.append(0)
        if yPredict[i] == ytest[i]:
            correct += 1

    Accuracy = correct*1.0/len(Xtest)

    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]
    #textDataSetsDirectoryFullPath = "/home/cabbage/Desktop/data_sets"
    #testFileDirectoryFullPath = "/home/cabbage/Desktop/data_sets/test_set"

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)
    #Xtrain, Xtest, ytrain, ytest = loadData("/home/cabbage/Desktop/data_sets")

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, vocabulary)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"

'''
Xtrain, Xtest, ytrain, ytest = loadData("/home/cabbage/Desktop/data_sets")
thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
'''
