# Starting code for CS6316/4501 HW3, Fall 2016
# By Weilin Xu

import math
import numpy as np
import random
from sklearn import svm, preprocessing
import csv


# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    global scaler

    def __init__(self):
        random.seed(0)

    def sperate_data(self, csv_fpath):
        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'native-country']
        # col_names_y = ['label']

        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']

        # load file with comma-delimited values
        data = np.genfromtxt(csv_fpath, dtype=str, delimiter=", ")
        lines = []
        num_data = []
        cat_data = []
        label = []
        for row in data:
            num_line = []
            cat_line = []
            # data without last column
            for i in range(0, row.size - 1):
                if col_names_x[i] in numerical_cols:
                    num_line.append(row[i])
                if col_names_x[i] in categorical_cols:
                    cat_line.append(row[i])
            # label with last column
            label.append(row[row.size - 1])

            # add one row
            num_data.append(num_line)
            cat_data.append(cat_line)
            lines.append(row)

        num_data = np.array(num_data)
        cat_data = np.array(cat_data)
        label = np.array(label)
        return num_data, cat_data, label

    def pre_num(self, num_data, is_train):
        if is_train:
            scaler = preprocessing.StandardScaler().fit(num_data)
        num_data = scaler.transform(num_data)
        return num_data

    def pre_cat(self, cat_data, is_train):
        tags = []
        tags.append(
            ['?', 'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
             'Never-worked'])
        tags.append(
            ['?', 'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
             '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
        tags.append(
            ['?', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
             'Married-AF-spouse'])
        tags.append(['?', 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                     'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                     'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
        tags.append(['?', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
        tags.append(['?', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        tags.append(['?', 'Female', 'Male'])
        tags.append(
            ['?', 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
             'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras',
             'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France',
             'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
             'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
             'Holand-Netherlands'])

        le = preprocessing.LabelEncoder()

        lencoders = []
        # remove '?'
        for i in range(len(cat_data[0])):
            # for tag in tags:
            lencoders.append(le.fit(tags[i]))
            cat_data[:, i] = lencoders[i].transform(cat_data[:, i])
            imp = preprocessing.Imputer(missing_values=0, strategy='most_frequent', axis=1)
            cat_data[:, i] = imp.fit_transform(cat_data[:, i])

        if is_train:
            n_values = []
            for i in range(len(tags)):
                n_values.append(len(tags[i]) - 1)
            ohenc = preprocessing.OneHotEncoder(n_values)

        # change from 1-41 to 0-40 ---remove '?'
        for row in range(len(cat_data)):
            cat_data[row] = [float(x) - 1 for x in cat_data[row]]

        cat_data = ohenc.fit_transform(cat_data).toarray()
        return cat_data

    def pre_label(self, labels, is_train):
        if is_train:
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)
        return labels

    def load_data(self, csv_fpath, is_train):

        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.
        num_data, cat_data, label = self.sperate_data(csv_fpath)

        num_data = self.pre_num(num_data, is_train)
        cat_data = self.pre_cat(cat_data, is_train)
        label = self.pre_label(label, is_train)

        x = np.c_[num_data, cat_data]
        return x, label

    def train_and_select_model(self, training_csv):
        x_train, y_train = self.load_data(training_csv, True)

        # 2. Select the best model with cross validation.
        # Attention: Write your own hyper-parameter candidates.
        param_set = [
            {'kernel': 'linear', 'C': 1, 'degree': 3, 'gamma': 'auto'},  # C
            {'kernel': 'poly', 'C': 1, 'degree': 1, 'gamma': 1},  # C, degree, gamma
            {'kernel': 'rbf', 'C': 1, 'degree': 3, 'gamma': 1},  # C, gamma
        ]

        best_score = 0
        for num_model in range(len(param_set)):
            param = param_set[num_model]
            trained_model = svm.SVC(C=param.get('C'), kernel=param.get('kernel'), degree=param.get('degree'),
                                    gamma=param.get('gamma'))

            norm1 = 0
            fold_num = 3
			print x_train[0]
            for num_test in range(fold_num):
                trainX, trainY, testX, testY = self.split_folds(x_train, y_train, fold_num, num_test)
                trained_model.fit(trainX, trainY)
                predictions = trained_model.predict(testX)
                diff = np.subtract(predictions, testY)
                norm1 += np.linalg.norm(diff, 1)
            temp_score = 100 - norm1 / len(y_train) * 100
            print(("Model was scored %.2f with " % temp_score) + param.get('kernel') + param.get('C') + param.get(
                'degree') + param.get('gamma'))

            if temp_score > best_score:
                best_model = trained_model
                best_score = temp_score

        return best_model, best_score

    def predict(self, test_csv, trained_model):
        x_test, _ = self.load_data(test_csv, False) # forgive y_test with _
        predictions = trained_model.predict(x_test)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')


if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print("The best model was scored %.2f" % cv_score)
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)
