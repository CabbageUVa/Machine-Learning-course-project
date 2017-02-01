#!/usr/bin/env python
#         0    1   2   3   4   5   6   7   8   9 Total
#Train 1194 1005 731 658 652 556 664 645 542 644 7291
# Test  359  264 198 166 200 160 170 147 166 177 2007

import sys
import csv
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA

train_label = []
train_data = []

test_real_label = []
test_data = []


def decision_tree(train, test):

    fid = open(train)
    tid = open(test)
    for line in fid:
        line = line.strip()
        m = line.split(' ')
        train_label.append(int(float(m[0])))
        train_data.append(m[1:])

    for line in tid:
        line = line.strip()
        m = line.split(' ')
        test_real_label.append(int(float(m[0])))
        test_data.append(m[1:])

    count = 0

    clf1 = tree.DecisionTreeClassifier()
    clf1 = clf1.fit(train_data, train_label)
    y1 = clf1.predict(test_data)
    acc1 = 0.0
    for i in range(2007):
        if int(float(y1[i])) == test_real_label[i]:
            count += 1
    acc1 = count * 1.0 / 2007

    clf1 = tree.DecisionTreeClassifier(criterion="entropy", max_features=20, random_state=10)

    clf1 = clf1.fit(train_data, train_label)
    y2 = clf1.predict(test_data)
    acc2 = 0.0
    count = 0
    for i in range(2007):
        if int(float(y2[i])) == test_real_label[i]:
            count += 1
    acc2 = count * 1.0 / 2007

    clf1 = tree.DecisionTreeClassifier(presort=True, random_state=5, class_weight="balanced")
    clf1 = clf1.fit(train_data, train_label)
    y3 = clf1.predict(test_data)
    acc3 = 0.0
    count = 0

    for i in range(2007):
        if int(float(y3[i])) == test_real_label[i]:
            count += 1
    acc3 = count * 1.0 / 2007

    y = y3
    #acc1 = 82.461      acc2 = 82.660       acc3 = 83.856
    return y

def knn(train, test):
    fid = open(train)
    tid = open(test)
    for line in fid:
        line = line.strip()
        m = line.split(' ')
        train_label.append(int(float(m[0])))
        train_data.append(m[1:])

    for line in tid:
        line = line.strip()
        m = line.split(' ')
        test_real_label.append(int(float(m[0])))
        test_data.append(m[1:])
    count = 0

    neigh = KNeighborsClassifier(n_neighbors = 1)
    neigh.fit(train_data, train_label)
    y1 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y1[i])) == test_real_label[i]:
            count += 1
    acc1 = count * 1.0 / 2007

    count = 0

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_data, train_label)
    y2 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y2[i])) == test_real_label[i]:
            count += 1
    acc2 = count * 1.0 / 2007

    count = 0

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_data, train_label)
    y3 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y3[i])) == test_real_label[i]:
            count += 1
    acc3 = count * 1.0 / 2007

    count = 0
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(train_data, train_label)
    y4 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y4[i])) == test_real_label[i]:
            count += 1
    acc4 = count * 1.0 / 2007

    count = 0
    neigh = KNeighborsClassifier(n_neighbors=200)
    neigh.fit(train_data, train_label)
    y5 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y5[i])) == test_real_label[i]:
            count += 1
    acc5 = count * 1.0 / 2007

    y = y2
    #acc1 = 94.369      acc2 = 94.469       acc3 = 93.572       acc4 = 91.778       acc5 = 82.810
    return y

def svm(train, test):
    fid = open(train)
    tid = open(test)
    for line in fid:
        line = line.strip()
        m = line.split(' ')
        train_label.append(int(float(m[0])))
        train_data.append(m[1:])

    for line in tid:
        line = line.strip()
        m = line.split(' ')
        test_real_label.append(int(float(m[0])))
        test_data.append(m[1:])

    param_set = [
        {'kernel': 'rbf', 'C': 100, 'degree': 3, 'gamma': 0.01},
        {'kernel': 'linear', 'C': 0.01, 'degree': 3, 'gamma': 0.01},
        {'kernel': 'poly', 'C': 1, 'degree': 1, 'gamma': 0.01}
    ]


    param = param_set[0]
    trained_model = SVC(C=param.get('C'), kernel=param.get('kernel'), degree=param.get('degree'),
                            gamma=param.get('gamma'))

    trained_model.fit(train_data, train_label)
    count = 0
    y1 = trained_model.predict(test_data)
    for i in range(2007):
        if int(float(y1[i])) == test_real_label[i]:
            count += 1
    acc1 = count * 1.0 / 2007

    param = param_set[1]
    trained_model = SVC(C=param.get('C'), kernel=param.get('kernel'), degree=param.get('degree'),
                            gamma=param.get('gamma'))

    trained_model.fit(train_data, train_label)
    count = 0
    y2 = trained_model.predict(test_data)
    for i in range(2007):
        if int(float(y2[i])) == test_real_label[i]:
            count += 1
    acc2 = count * 1.0 / 2007

    param = param_set[2]
    trained_model = SVC(C=param.get('C'), kernel=param.get('kernel'), degree=param.get('degree'),
                            gamma=param.get('gamma'))

    trained_model.fit(train_data, train_label)
    count = 0
    y3 = trained_model.predict(test_data)
    for i in range(2007):
        if int(float(y3[i])) == test_real_label[i]:
            count += 1
    acc3 = count * 1.0 / 2007

    #acc1 = 95.316
    #acc2 = 93.173
    #acc3 = 93.173
    y = y1
    return y

def pca_knn(train, test):
    fid = open(train)
    tid = open(test)
    for line in fid:
        line = line.strip()
        m = [int(float(x)) for x in line.split(' ')]
        train_label.append(m[0])
        train_data.append(m[1:])

    for line in tid:
        line = line.strip()
        m = [int(float(x)) for x in line.split(' ')]
        test_real_label.append(m[0])
        test_data.append(m[1:])


    pca = RandomizedPCA(n_components=5)
    pca.fit(train_data)

    train_data_5 = pca.transform(train_data)
    test_data_5 = pca.transform(test_data)
    count = 0

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_data_5, train_label)
    y1 = neigh.predict(test_data_5)

    for i in range(2007):
        if int(float(y1[i])) == test_real_label[i]:
            count += 1
    acc1 = count * 1.0 / 2007

    pca = RandomizedPCA(n_components=20)
    pca.fit(train_data)

    train_data_20 = pca.transform(train_data)
    test_data_20 = pca.transform(test_data)
    count = 0

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_data_20, train_label)
    y2 = neigh.predict(test_data_20)

    for i in range(2007):
        if int(float(y2[i])) == test_real_label[i]:
            count += 1
    acc2 = count * 1.0 / 2007

    y = y2
    #acc1 = 0.7777      acc2 = 0.9337
    return y

def pca_svm(train, test):
    fid = open(train)
    tid = open(test)
    for line in fid:
        line = line.strip()
        m = [int(float(x)) for x in line.split(' ')]
        train_label.append(m[0])
        train_data.append(m[1:])

    for line in tid:
        line = line.strip()
        m = [int(float(x)) for x in line.split(' ')]
        test_real_label.append(m[0])
        test_data.append(m[1:])

    pca = RandomizedPCA(n_components=5)
    pca.fit(train_data)

    train_data_5 = pca.transform(train_data)
    test_data_5 = pca.transform(test_data)
    trained_model = SVC(C=100, kernel='rbf', degree=3,
                        gamma=0.01)
    trained_model.fit(train_data_5, train_label)
    count = 0
    y1 = trained_model.predict(test_data_5)
    for i in range(2007):
        if int(float(y1[i])) == test_real_label[i]:
            count += 1
    acc1 = count * 1.0 / 2007

    pca = RandomizedPCA(n_components=20)
    pca.fit(train_data)

    train_data_20 = pca.transform(train_data)
    test_data_20 = pca.transform(test_data)
    trained_model = SVC(C=100, kernel='rbf', degree=3,
                        gamma=0.01)
    trained_model.fit(train_data_20, train_label)
    count = 0
    y2 = trained_model.predict(test_data_20)
    for i in range(2007):
        if int(float(y2[i])) == test_real_label[i]:
            count += 1
    acc2 = count * 1.0 / 2007

    y = y2
    #acc1 = 0.7997      acc2 = 0.9417
    return y

if __name__ == '__main__':
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]

    if model == "dtree":
        print(decision_tree(train, test))
    elif model == "knn":
        print(knn(train, test))
    elif model == "svm":
        print(svm(train, test))
    elif model == "pcaknn":
        print(pca_knn(train, test))
    elif model == "pcasvm":
        print(pca_svm(train, test))
    else:
        print("Invalid method selected!")
