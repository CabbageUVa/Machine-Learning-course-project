########################################
# Written with Python 2.7
# Modified on Sep. 23th 2017
# UVa CS-6316 Machine Learning HW2 Q1
# Author: Bicheng Fang (bf5ef)
########################################

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(addr):      # Read DataSet From Q1data.txt, and Draw a Figure

    dataBuff = np.array([[0,0,0]], dtype = float).reshape(3,1)         # initial the data buff
    dataBuff = np.delete(dataBuff, 0, 1)
    file = open(addr)
    line = file.readline()

    while line:
        arr = line.split('\t')    #split numbers
        tempNum = np.array([[arr[0],arr[1],arr[2]]], dtype = float).reshape(3,1)
        dataBuff = np.c_[dataBuff, tempNum]     #append a column
        line = file.readline()      #read the next line

    plt.scatter(dataBuff[1], dataBuff[2])   #draw a pic
    #plt.show()
    plt.savefig("Q1_1.png")
    return np.r_[dataBuff[0],dataBuff[1]], dataBuff[2]


def standRegres(xVal, yVal):

    X_T = xVal #X transpose
    X   = np.transpose(X_T)
    Y= yVal.reshape(200,1)

    theta = (np.linalg.inv(np.dot( X_T, X)).dot(X_T)).dot(Y)

    plt.scatter(xVal[1], Y)  # draw a pic

    testPointX = np.linspace(-0.5, 1.5, 100)

    plt.plot(testPointX, theta[0] + theta[1]*testPointX)  # draw a regression line
    #plt.show()
    plt.savefig("Q1_2.png")
    print theta
    return theta


def polyRegres(xVal, yVal):

    X_T = xVal                          ##########################
    Xpow2 = X_T[1]**2                   #
    Xpow2 = Xpow2.reshape(1,200)        #   Deal with pow(X1, 2)
    X_T = np.r_[X_T, Xpow2]             #   and add this row into
    X = np.transpose(X_T)               #   the original array
    Y = yVal.reshape(200,1)             #
					##########################


    theta = (np.linalg.inv(np.dot(X_T, X)).dot(X_T)).dot(Y)

    plt.scatter(xVal[1], yVal)  # draw a pic

    testPointX = np.linspace(-1, 2, 200)

    plt.plot(testPointX, theta[0] + theta[1] * testPointX + theta[2] * (testPointX**2))  # draw a regression line
    plt.savefig("Q1_3.png")

    print theta
    return theta


xVal, yVal = loadDataSet("Q2data.txt")     #dataSet records X0, X1 and Y, read from Q1data.txt

# Reshape the xVal and yVal to row vectors
xVal = xVal.reshape(2,200)
yVal = yVal.reshape(1,200)

theta = standRegres(xVal, yVal)            #theta is a two dimontional metrix, representing the result conluded from normal equition
theta_Poly = polyRegres(xVal, yVal)
