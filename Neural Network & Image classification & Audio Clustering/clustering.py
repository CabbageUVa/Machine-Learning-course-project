#!/usr/bin/python

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
#Your code here

def loadData(fileDj):
    data = []
    fid = open(fileDj)
    for line in fid:
        line = line.strip()
        m = [float(x) for x in line.split(' ')]
        data.append(m)


    return data

## K-means functions 

def getInitialCentroids(X, k):
    initialCentroids = []

    for i in range(k):
        index = random.randint(0, len(X))
        initialCentroids.append(X[index])

    #Your code here
    return initialCentroids


def visualizeClusters(clusters):

    for i in range(len(clusters)):
        clusters[i] = np.array(clusters[i])

    plt.plot(clusters[0][:,0], clusters[0][:,1], 'rs', clusters[1][:,0], clusters[1][:,1], 'bs')
    plt.show()
    return

def has_converged(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 100
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

def euclidean_dist(data, centroids, clusters):
    centroids = np.array(centroids)
    for instance in data:
        instance = np.array(instance)

        mu_index = min([(i[0], np.linalg.norm(instance - centroids[i[0]])) \
                        for i in enumerate(centroids)], key=lambda t: t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters


def kmeans(X, k, maxIter=1000):

    centroids = getInitialCentroids(X,k)

    old_centroids = [[] for i in range(k)]

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(X, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1

    visualizeClusters(clusters)

    return clusters

def kmeans_(X, k, maxIter=1000):

    centroids = getInitialCentroids(X,k)

    old_centroids = [[] for i in range(k)]

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(X, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1

    #visualizeClusters(clusters)

    return clusters


def Func(clusters):
    center = []
    for i in range(len(clusters)):
        center.append(clusters[i][0])

    distSum = 0

    for i in range(len(clusters)):
        for j in range(1, len(clusters[i])):
            distSum += np.linalg.norm(center[i] - clusters[i][j])

    return distSum

def kneeFinding(X,kList):
    obj = []

    for i in kList:
        obj.append(Func(kmeans_(X, i)))

    plt.plot(range(1,7), obj)
    plt.show()

    return

def purity(X, clusters):
    purities = []
    #Your code
    for i in range(2):
        count = 0
        for idx in range(len(clusters[i])):
            if(int(clusters[i][idx][2]) == 1):
                count += 1

        purity = count*1.0 / len(clusters[i])
        if purity > 0.5:
            purities.append(purity)
        else:
            purities.append(1-purity)

    #<type 'list'>: [0.9724249797242498, 0.999000999000999]
    return purities

'''

## GMM functions 

#calculate the initial covariance matrix
#covType: diag, full
def getInitialsGMM(X,k,covType):
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])-1):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)

    initialClusters = {}
    #Your code here
    return initialClusters


def calcLogLikelihood(X,clusters,k):
    loglikelihood = 0
    #Your code here
    return loglikelihood

#E-step
def updateEStep(X,clusters,k):
    EMatrix = []
    #Your code here
    return EMatrix

#M-step
def updateMStep(X,clusters,EMatrix):
    #Your code here
    return clusters

def visualizeClustersGMM(X,labels,clusters,covType):
    #Your code here


def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clustersGMM = getInitialsGMM(X,k,covType)
    labels = []
    #Your code here
    visualizeClustersGMM(X,labels,clustersGMM,covType)
    return labels,clustersGMM


def purityGMM(X, clusters, labels):
    purities = []
    #Your code here
    return purities


'''

def main():
    #######dataset path
    #datadir = sys.argv[1]
    datadir = ''
    pathDataset1 = datadir+'humanData.txt'
    #pathDataset2 = datadir+'/audioData.txt'
    dataset1 = loadData(pathDataset1)
    #dataset2 = loadData(pathDataset2)


    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    clusters = kmeans(dataset1, 2, maxIter=1000)
    purity(dataset1,clusters)
'''
    #Q7
    labels11,clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
    labels12,clustersGMM12 = gmmCluster(dataset1, 2, 'full')

    #Q8
    labels21,clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
    labels22,clustersGMM22 = gmmCluster(dataset2, 2, 'full')

    #Q9
    purities11 = purityGMM(dataset1, clustersGMM11, labels11)
    purities12 = purityGMM(dataset1, clustersGMM12, labels12)
    purities21 = purityGMM(dataset2, clustersGMM21, labels21)
    purities22 = purityGMM(dataset2, clustersGMM22, labels22)
'''
if __name__ == "__main__":
    main()