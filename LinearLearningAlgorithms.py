import os

import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC


class Classifier(object):

    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged=False, eta=1.5, alpha=1.1):

        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        features = {feature for xi in x_train for feature in xi.keys()}
        # variable to count total # of mistakes
        self.correct_classifications = 0

        if algorithm == 'Perceptron':
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            mistakes = 0  # counts number of mistakes
            counter = 1  # to keep weights of different models

            if averaged:
                # aggregate weights and bias for averaged perceptron

                self.u, self.u['bias'] = {feature: 0.0 for feature in features}, 0.0

            # Iterate over the training data n times
            for curr_iter in range(iterations):
                # Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a missclassification
                    if yi != y_hat:
                        mistakes = mistakes + 1

                        # Update for Normal Perceptron
                        if not averaged:
                            for feature, value in xi.items():
                                self.w[feature] = self.w[
                                    feature] + yi * eta * value
                            self.w['bias'] = self.w['bias'] + yi * eta

                        # Update for Averaged Perceptron
                        if averaged:
                            for feature, value in xi.items():
                                self.w[feature] = self.w[feature] + \
                                    yi * eta * value  # update weights
                                self.u[feature] = self.u[feature] + yi * \
                                    eta * value * counter  # update agg. weights
                            self.w['bias'] = self.w['bias'] + \
                                yi * eta  # update bias
                            self.u['bias'] = self.u['bias'] + yi * \
                                eta * counter  # update agg. bias

                    counter = counter + 1  # update counter every step

            self.correct_classifications = counter - mistakes

            if averaged:
                for feature, value in self.u.items():
                    self.u[feature] = (value / counter)  # get u * (1/c)

                for feature, value in self.w.items():
                    self.w[feature] = value - \
                        self.u[feature]  # w_avg = w - u(1/c)

        if algorithm == "Winnow":
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 1.0 for feature in features}, -(len(features))
            counter = 0  # keeps the count of correct classifications
            total = 0  # keeps count of total examples seen

            if averaged:
                # Aggregate weights, bias
                self.u, self.u['bias'] = {feature: 1.0 for feature in features}, -(len(features))

            for curr_iter in range(iterations):
                # Check each training example
                for i in range(len(x_train)):

                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # if we missclassify xi using weight vector w
                    if yi != y_hat:

                        # update for normal Winnow
                        if not averaged:
                            for feature, value in xi.items():
                                update_term = alpha ** (yi * value)
                                self.w[feature] = self.w[feature] * \
                                    update_term  # update weights

                        # update for Averaged Winnow
                        if averaged:

                            # A mistake was made. Add a weighted copy of
                            # current weight vector to aggregate weights
                            for feature, value in self.u.items():
                                if feature is not 'bias':
                                    self.u[feature] = self.u[feature] + \
                                        counter * self.w[feature]

                            self.u['bias'] = self.u['bias'] + counter * \
                                self.w['bias']  # add weighted bias

                            for feature, value in xi.items():
                                update_term = alpha ** (yi * value)
                                self.w[feature] = self.w[feature] * \
                                    update_term  # update weights

                        self.correct_classifications = self.correct_classifications + counter
                        counter = 0  # end of missclassify

                    total = total + 1
                    counter = counter + 1

            if averaged:

                for feature, value in self.u.items():
                    if feature is not 'bias':
                        self.u[feature] = self.u[feature] + \
                            counter * self.w[feature]

                self.u['bias'] = self.u['bias'] + counter * \
                    self.w['bias']  # add weighted bias

                for feature, value in self.w.items():
                    self.w[feature] = self.u[feature]

        if algorithm == "SVM":
            # variables are to store model and svm.score() accuracy
            self.clf = LinearSVC()
            self.svm_accuracy = 0

        if algorithm == "Adagrad":

            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Initialize gradient vector and a dictionary that will keep sum of
            # gradient squares.
            self.g, self.g['bias'] = {feature: 0.0 for feature in features}, 0.0
            self.G, self.G['bias'] = {feature: 0.0 for feature in features}, 0.0

            # variables for Averaged Version
            counter = 0  # keeps the count of current correct classifications
            total = 1  # keeps count of total examples seen

            if averaged:
                # Aggregate weights, bias
                self.u, self.u['bias'] = {feature: 0.0 for feature in features}, 0.0

            # Iterate over the training data n times
            for current_iter in range(iterations):
                # Check each training example
                for i in range(len(x_train)):

                    xi, yi = x_train[i], y_train[i]

                    # notice we want (wx + bias) not 1 or -1.
                    wx_and_bias = self.predict_adagrad(xi)

                    # correct classification
                    if (yi * wx_and_bias) > 1:
                        for feature, value in self.g.items():
                            self.g[feature] = 0
                        counter = counter + 1  # increment number of correct classifications

                    # A mistake was made. So we update
                    else:

                        if not averaged:
                            # update gradient vector
                            for feature, value in xi.items():
                                self.g[feature] = -(value * yi)

                            self.g['bias'] = -yi  # update bias

                            # update sums of gradient squares
                            for feature, value in self.g.items():
                                self.G[feature] = self.G[
                                    feature] + (self.g[feature] ** 2)

                            # update weight vector according to update rule
                            # specified.
                            for feature, value in xi.items():
                                sqrt_G = math.sqrt((self.G[feature]))
                                self.w[feature] = self.w[feature] - \
                                    ((eta * self.g[feature]) / sqrt_G)

                            sqrt_bias_G = math.sqrt((self.G['bias']))
                            self.w['bias'] = self.w['bias'] - \
                                (eta * self.g['bias'] / sqrt_bias_G)

                            self.correct_classifications = self.correct_classifications + counter
                            counter = 0  # reset number of correct classifications

                        if averaged:
                            for feature, value in xi.items():
                                self.g[feature] = -(value * yi)

                            self.g['bias'] = -yi  # update bias

                            # update sums of gradient squares
                            for feature, value in self.g.items():
                                self.G[feature] = self.G[
                                    feature] + (self.g[feature] ** 2)

                            # update weight vector according to update rule
                            # specified.
                            for feature, value in xi.items():
                                sqrt_G = math.sqrt((self.G[feature]))
                                self.w[feature] = self.w[feature] - \
                                    ((eta * self.g[feature]) / sqrt_G)
                                self.u[feature] = self.u[feature] - \
                                    ((eta * self.g[feature] * total) / sqrt_G)
                            sqrt_bias_G = math.sqrt((self.G['bias']))
                            self.w['bias'] = self.w['bias'] - \
                                (eta * self.g['bias'] / sqrt_bias_G)
                            self.u['bias'] = self.u['bias'] - \
                                (eta * self.g['bias'] * total / sqrt_bias_G)

                            self.correct_classifications = self.correct_classifications + counter
                            counter = 0  # reset number of correct classifications

                    total = total + 1  # increment number of seen examples

            if averaged:
                for feature, value in self.u.items():
                    self.u[feature] = (value / total)  # get u * (1/c)

                for feature, value in self.w.items():
                    self.w[feature] = value - \
                        self.u[feature]  # w_avg = w - u(1/c)

    def predict(self, x):
        s = sum([self.w[feature] * value for feature,
                 value in x.items()]) + self.w['bias']

        return 1 if s > 0 else -1

    # For hinge loss calculations
    def predict_adagrad(self, x):
        s = sum([self.w[feature] * value for feature,
                 value in x.items()]) + self.w['bias']
        return s

    # For transforming Data with DictVectorizer() and calculating accuracy of
    # SVM model
    def get_svm_accuracy(self, x_train, y_train, x_dev, y_dev):

        v = DictVectorizer()
        svm_train_x = v.fit_transform(x_train)

        svm_train_y = np.asarray(y_train)
        self.clf.fit(svm_train_x, svm_train_y)
        x_svm = v.transform(x_dev)
        y_svm = np.asarray(y_dev)

        self.svm_accuracy = (self.clf.score(x_svm, y_svm) * 100)

    # Function to train a SVM Model and transform our test data into required format.
    # The return values of this function are utilized by two functions that
    # get predictions from test data.
    def get_svm_predictions(self, x_train, y_train, x_dev):
        v = DictVectorizer()
        svm_train_x = v.fit_transform(x_train)

        svm_train_y = np.asarray(y_train)

        self.clf.fit(svm_train_x, svm_train_y)
        x_svm = v.transform(x_dev)

        return [self.clf, x_svm]


# Tries alpha values for Winnow Algorithm and plots their Dev Accuracies.
def tune_alpha_winnow(x_train, y_train, x_test, y_test):
    global syn_sparse_train_x, syn_sparse_train_y, syn_sparse_dev_x, syn_sparse_dev_y

    parameter_values = [1.1, 1.01, 1.005, 1.0005, 1.0001]

    xi = [i for i in range(0, len(parameter_values))]

    parameter_scores = []

    for param in parameter_values:
        # We plot different alpha values with their dev accuracies
        w_tune = Classifier('Winnow', x_train, y_train, alpha=param)
        acc = sum(
            [1 for i in range(len(y_test)) if
             w_tune.predict(x_test[i]) == y_test[i]]) / len(
            y_test) * 100
        parameter_scores.append(acc)

    plt.plot(xi, parameter_scores, marker='o',
             linestyle='--', color='r', label='alpha')

    plt.xlabel('Alpha value for Averaged Winnow')
    plt.ylabel('Accuracy of Dev Data in Percentage')
    plt.xticks(xi, parameter_values)
    plt.legend()

    plt.show()


# Tries learning rate values for Adagrad Algorithm and plots their Dev
# Accuracies.
def tune_eta_adagrad(x_train, y_train, x_test, y_test):
    global syn_sparse_train_x, syn_sparse_train_y, syn_sparse_dev_x, syn_sparse_dev_y

    parameter_values = [1.5, 0.25, 0.03, 0.005, 0.001]

    xi = [i for i in range(0, len(parameter_values))]

    parameter_scores = []

    for param in parameter_values:
        # We plot different eta values with their dev accuracies .
        a = Classifier('Adagrad', x_train, y_train, eta=param)
        acc = sum(
            [1 for i in range(len(y_test)) if
             a.predict(x_test[i]) == y_test[i]]) / len(
            y_test) * 100
        parameter_scores.append(acc)

    plt.plot(xi, parameter_scores, marker='o',
             linestyle='--', color='r', label='eta')

    plt.xlabel('Eta Value for Adagrad ')
    plt.ylabel('Accuracy on Dev Data')
    plt.xticks(xi, parameter_values)
    plt.legend()

    plt.show()


# Plots learning curve of SVM, Perceptron, Winnow, Adagrad and their
# averaged versions on 11 different training data sizes
def plot_learning_curves(x_train, y_train, x_test, y_test):
    global syn_sparse_train_x, syn_sparse_train_y, syn_sparse_dev_x, syn_sparse_dev_y, syn_dense_train_x, syn_dense_train_y, syn_dense_dev_x, syn_dense_dev_y

    training_set_sizes = [500, 1000, 1500, 2000,
                          2500, 3000, 3500, 4000, 4500, 5000, 50000]
    xi = [i for i in range(0, len(training_set_sizes))]

    models = [("Perceptron", False), ("Perceptron", True), ("Winnow", False), ("Winnow", True), ("Adagrad", False),
              ("Adagrad", True), ('SVM', False)]

    for model in models:
        accuracy_scores = []
        for train_size in training_set_sizes:

            if train_size == 50000:
                updated_train_set = x_train
                updated_label_set = y_train

            else:
                updated_train_set = x_train[:train_size]
                updated_label_set = y_train[:train_size]

            if model[0] is not 'SVM':
                current_model = Classifier(model[0], updated_train_set, updated_label_set, iterations=10,
                                           averaged=model[1])
                acc = sum(
                    [1 for i in range(len(y_test)) if
                     current_model.predict(x_test[i]) == y_test[i]]) / len(
                    y_test) * 100
                accuracy_scores.append(acc)
            else:
                current_model = Classifier(model[0], updated_train_set, updated_label_set,
                                           averaged=model[1])
                current_model.get_svm_accuracy(
                    updated_train_set, updated_label_set, x_test, y_test)

                accuracy_scores.append(current_model.svm_accuracy)

        plt.xlabel('Training Set Sizes')
        plt.ylabel('Accuracy of Dev Data in % ')

        plt.xticks(xi, training_set_sizes)

        if model[1]:
            plt.plot(xi, accuracy_scores, marker='.',
                     linestyle='-', label='Averaged ' + model[0])
        else:
            plt.plot(xi, accuracy_scores, marker='.',
                     linestyle='-', label=model[0])

    plt.legend()
    plt.show()


# Writes predictions of Averaged Perceptron and SVM into text files for
# CoNLL and Enron Test Data
def get_predictions_from_real_test_data(test_data, test_data_name, model_name, model):
    if model_name == 'Perceptron':
        openfile = open('p' + '-' + test_data_name + '.txt', 'w')
        for i in range(len(test_data)):
            if (model.predict(test_data[i])) == 1:
                openfile.write('I' + "\n")
            else:
                openfile.write('O' + "\n")
        openfile.close()
    else:
        openfile = open('svm' + '-' + test_data_name + '.txt', 'w')
        print(test_data.shape[0])
        for i in range((test_data.shape[0])):
            if (model.predict(test_data[i])) == 1:
                openfile.write('I' + "\n")
            else:
                openfile.write('O' + "\n")
        openfile.close()


# Writes predictions of Averaged Perceptron and SVM into text files for
# Sparse and Dense Synthehic Test Data
def get_predictions_from_syn_test_data(test_data, test_data_name, model_name, model):
    if model_name == 'Perceptron':
        openfile = open('p' + '-' + test_data_name + '.txt', 'w')
        for i in range(len(test_data)):
            openfile.write(str(model.predict(test_data[i])) + "\n")

        openfile.close()
    # if we are using SVM model
    else:
        openfile = open('svm' + '-' + test_data_name + '.txt', 'w')
        for i in range((test_data.shape[0])):
            openfile.write(str(model.predict(test_data[i])[0]) + "\n")
        openfile.close()


# Parse the real-world data to generate features,
# Returns a list of tuple lists
def parse_real_data(path):
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path + filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data


# Returns a list of labels
def parse_synthetic_labels(path):
    # List of tuples for each sentence
    labels = []
    with open(path + 'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


# Returns a list of features
def parse_synthetic_data(path):
    # List of tuples for each sentence
    data = []
    with open(path + 'x.txt') as file:
        features = []
        for line in file:
            # print('Line:', line)
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data


if __name__ == '__main__':
    print('Loading data...')
    # Load data from folders.
    # Real world data - lists of tuple lists

    news_train_data = parse_real_data('Data/Real-World/CoNLL/train/')
    news_dev_data = parse_real_data('Data/Real-World/CoNLL/dev/')
    news_test_data = parse_real_data('Data/Real-World/CoNLL/test/')
    email_dev_data = parse_real_data('Data/Real-World/Enron/dev/')
    email_test_data = parse_real_data('Data/Real-World/Enron/test/')

    # #Load dense synthetic data
    syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/train/')
    syn_dense_train_labels = parse_synthetic_labels(
        'Data/Synthetic/Dense/train/')
    syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/dev/')
    syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/dev/')

    # Load sparse synthetic data
    syn_sparse_train_data = parse_synthetic_data(
        'Data/Synthetic/Sparse/train/')
    syn_sparse_train_labels = parse_synthetic_labels(
        'Data/Synthetic/Sparse/train/')
    syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/dev/')
    syn_sparse_dev_labels = parse_synthetic_labels(
        'Data/Synthetic/Sparse/dev/')

    # Load test data for synthetic data
    syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/test/')
    syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/test/')

    # Convert to sparse dictionary representations.
    print('Converting Synthetic data...')
    syn_dense_train = zip(*[({'x' + str(i): syn_dense_train_data[j][i]
                              for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1},
                             syn_dense_train_labels[j])
                            for j in range(len(syn_dense_train_data))])
    syn_dense_train_x, syn_dense_train_y = syn_dense_train

    syn_dense_dev = zip(*[({'x' + str(i): syn_dense_dev_data[j][i]
                            for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1},
                           syn_dense_dev_labels[j])
                          for j in range(len(syn_dense_dev_data))])
    syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev

    syn_sparse_train = zip(*[({'x' + str(i): syn_sparse_train_data[j][i]
                               for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1},
                              syn_sparse_train_labels[j])
                             for j in range(len(syn_sparse_train_data))])
    syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train
    syn_sparse_dev = zip(*[({'x' + str(i): syn_sparse_dev_data[j][i]
                             for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1},
                            syn_sparse_dev_labels[j])
                           for j in range(len(syn_sparse_dev_data))])
    syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev

    # Convert sparse and dense test data to their dictionary representations
    # so we can test them.
    syn_sparse_test_x = *({'x' + str(i): syn_sparse_test_data[j][i]
                           for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1}
                          for j in range(len(syn_sparse_test_data))),

    syn_dense_test_x = *({'x' + str(i): syn_dense_test_data[j][i]
                          for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1}
                         for j in range(len(syn_dense_test_data))),

    # Feature extraction. Modified to extract seven features instead of two.
    print('Extracting features from real-world data...')
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentences in news_train_data:
        padded = sentences[:]
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))

        for i in range(1, len(padded) - 1):

            news_train_y.append(1 if padded[i][1] == 'I' else -1)

            feat_m1 = 'w-1=' + str(padded[i - 1][0])
            feat_p1 = 'w+1=' + str(padded[i + 1][0])
            train_feats = [feat_m1, feat_p1]

            if i - 2 > -1:
                feat_m2 = 'w-2=' + str(padded[i - 2][0])
                train_feats.insert(0, feat_m2)
            if i + 2 < len(padded):
                feat_p2 = 'w+2=' + str(padded[i + 2][0])
                train_feats.append(feat_p2)

            if i - 2 > -1:
                feat_m2_m1 = 'w-2&w-1=' + \
                    str(padded[i - 2][0]) + " " + str(padded[i - 1][0])
                train_feats.append(feat_m2_m1)

            if i + 2 < len(padded):
                feat_p1_p2 = 'w+1&w+2=' + \
                    str(padded[i + 1][0]) + " " + str(padded[i + 2][0])
                train_feats.append(feat_p1_p2)

            feat_m1_p1 = 'w-1&w+1=' + \
                str(padded[i - 1][0]) + " " + str(padded[i + 1][0])
            train_feats.append(feat_m1_p1)

            train_features.update(train_feats)
            train_feats = {feature: 1 for feature in train_feats}
            news_train_x.append(train_feats)

    news_dev_y = []
    news_dev_x = []
    for sentences in news_dev_data:
        padded = sentences[:]
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))

        for i in range(1, len(padded) - 1):
            news_dev_y.append(1 if padded[i][1] == 'I' else -1)

            feat_m1 = 'w-1=' + str(padded[i - 1][0])
            feat_p1 = 'w+1=' + str(padded[i + 1][0])
            dev_feats = [feat_m1, feat_p1]

            if i - 2 > -1:
                feat_m2 = 'w-2=' + str(padded[i - 2][0])
                dev_feats.insert(0, feat_m2)
            if i + 2 < len(padded):
                feat_p2 = 'w+2=' + str(padded[i + 2][0])
                dev_feats.append(feat_p2)

            if i - 2 > -1:
                feat_m2_m1 = 'w-2&w-1=' + \
                    str(padded[i - 2][0]) + " " + str(padded[i - 1][0])
                dev_feats.append(feat_m2_m1)

            if i + 2 < len(padded):
                feat_p1_p2 = 'w+1&w+2=' + \
                    str(padded[i + 1][0]) + " " + str(padded[i + 2][0])
                dev_feats.append(feat_p1_p2)

            feat_m1_p1 = 'w-1&w+1=' + \
                str(padded[i - 1][0]) + " " + str(padded[i + 1][0])
            dev_feats.append(feat_m1_p1)

            dev_feats = {feature: 1 for feature in dev_feats if feature in train_features}
            news_dev_x.append(dev_feats)

    # Feature extraction from Enron Dev Data. Same code extracting features
    # from CoNLL Dev Data
    email_dev_y = []
    email_dev_x = []

    for sentences in email_dev_data:
        padded = sentences[:]
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))

        for i in range(1, len(padded) - 1):
            email_dev_y.append(1 if padded[i][1] == 'I' else -1)

            feat_m1 = 'w-1=' + str(padded[i - 1][0])
            feat_p1 = 'w+1=' + str(padded[i + 1][0])
            email_dev_feats = [feat_m1, feat_p1]

            if i - 2 > -1:
                feat_m2 = 'w-2=' + str(padded[i - 2][0])
                email_dev_feats.insert(0, feat_m2)
            if i + 2 < len(padded):
                feat_p2 = 'w+2=' + str(padded[i + 2][0])
                email_dev_feats.append(feat_p2)

            if i - 2 > -1:
                feat_m2_m1 = 'w-2&w-1=' + \
                    str(padded[i - 2][0]) + " " + str(padded[i - 1][0])
                email_dev_feats.append(feat_m2_m1)

            if i + 2 < len(padded):
                feat_p1_p2 = 'w+1&w+2=' + \
                    str(padded[i + 1][0]) + " " + str(padded[i + 2][0])
                email_dev_feats.append(feat_p1_p2)

            feat_m1_p1 = 'w-1&w+1=' + \
                str(padded[i - 1][0]) + " " + str(padded[i + 1][0])
            email_dev_feats.append(feat_m1_p1)

            email_dev_feats = {feature: 1 for feature in email_dev_feats if feature in train_features}
            email_dev_x.append(email_dev_feats)

    # Transform news_test_data into vector form so we can test it
    news_test_x = []
    for sentences in news_test_data:

        padded = []
        for word in sentences:
            formatted_string = ' '.join(map(str, word))
            padded.append(formatted_string)

        padded.insert(0, ('SSS'))
        padded.append(('EEE'))

        for i in range(1, len(padded) - 1):

            feat_m1 = 'w-1=' + str(padded[i - 1])
            feat_p1 = 'w+1=' + (str(padded[i + 1]))
            feats = [feat_m1, feat_p1]

            if i - 2 > -1:
                feat_m2 = 'w-2=' + str(padded[i - 2])
                feats.insert(0, feat_m2)
            if i + 2 < len(padded):
                feat_p2 = 'w+2=' + str(padded[i + 2])
                feats.append(feat_p2)

            if i - 2 > -1:
                feat_m2_m1 = 'w-2&w-1=' + \
                    str(padded[i - 2]) + " " + str(padded[i - 1])
                feats.append(feat_m2_m1)

            if i + 2 < len(padded):
                feat_p1_p2 = 'w+1&w+2=' + \
                    str(padded[i + 1]) + " " + str(padded[i + 2])
                feats.append(feat_p1_p2)

            feat_m1_p1 = 'w-1&w+1=' + \
                str(padded[i - 1]) + " " + str(padded[i + 1])
            feats.append(feat_m1_p1)
            feats = {feature: 1 for feature in feats if feature in train_features}
            news_test_x.append(feats)

    # Transform email_test_data into vector form so we can test it
    email_test_x = []
    for sentences in email_test_data:

        padded = []
        for word in sentences:
            formatted_string = ' '.join(map(str, word))
            padded.append(formatted_string)

        padded.insert(0, ('SSS'))
        padded.append(('EEE'))

        for i in range(1, len(padded) - 1):

            feat_m1 = 'w-1=' + str(padded[i - 1])
            feat_p1 = 'w+1=' + (str(padded[i + 1]))
            feats = [feat_m1, feat_p1]

            if i - 2 > -1:
                feat_m2 = 'w-2=' + str(padded[i - 2])
                feats.insert(0, feat_m2)
            if i + 2 < len(padded):
                feat_p2 = 'w+2=' + str(padded[i + 2])
                feats.append(feat_p2)

            if i - 2 > -1:
                feat_m2_m1 = 'w-2&w-1=' + \
                    str(padded[i - 2]) + " " + str(padded[i - 1])
                feats.append(feat_m2_m1)

            if i + 2 < len(padded):
                feat_p1_p2 = 'w+1&w+2=' + \
                    str(padded[i + 1]) + " " + str(padded[i + 2])
                feats.append(feat_p1_p2)

            feat_m1_p1 = 'w-1&w+1=' + \
                str(padded[i - 1]) + " " + str(padded[i + 1])
            feats.append(feat_m1_p1)
            feats = {feature: 1 for feature in feats if feature in train_features}
            email_test_x.append(feats)

    # Get Average Perceptron Accuracy on CoNLL Dev Data

    avg_p_news = Classifier('Perceptron', news_train_x,
                            news_train_y, iterations=10, averaged=True)
    avg_p_news_accuracy = sum([1 for i in range(len(news_dev_y))
                               if avg_p_news.predict(news_dev_x[i]) == news_dev_y[i]]) / len(news_dev_y) * 100
    print('News Dev Accuracy for Average Perceptron:', avg_p_news_accuracy)

    # Get Average Perceptron Accuracy on Email (Enron) Dev Data
    avg_p_email_accuracy = sum([1 for i in range(len(email_dev_y))
                                if avg_p_news.predict(email_dev_x[i])
                                == email_dev_y[i]]) / len(email_dev_y) * 100
    print('Email (Enron) Dev Accuracy for Average Perceptron:', avg_p_email_accuracy)

    # Get SVM Accuracy on CoNLL Dev Data

    svm_news = Classifier('SVM', news_train_x, news_train_y)
    svm_news.get_svm_accuracy(
        news_train_x, news_train_y, news_dev_x, news_dev_y)
    svm_news_accuracy = svm_news.svm_accuracy
    print('News Dev Accuracy for SVM:', svm_news_accuracy)

    # Get SVM Accuracy on Enron (Email) Dev Data

    svm_email = Classifier('SVM', news_train_x, news_train_y)
    svm_email.get_svm_accuracy(
        news_train_x, news_train_y, email_dev_x, email_dev_y)
    svm_email_accuracy = svm_email.svm_accuracy
    print('Email (Enron) Dev Accuracy for SVM:', svm_email_accuracy)

    # Print Accuracies of all 7 models on Dense and Sparse Synthetic
    # Development Data

    print('\nSVM Accuracy For Sparse and Dense Data')
    svm_sparse = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y)
    svm_sparse.get_svm_accuracy(
        syn_sparse_train_x, syn_sparse_train_y, syn_sparse_dev_x, syn_sparse_dev_y)
    print('Sparse Synthetic Dev Accuracy for SVM:', svm_sparse.svm_accuracy)

    svm_dense = Classifier('SVM', syn_dense_train_x, syn_dense_train_y)
    svm_dense.get_svm_accuracy(
        syn_dense_train_x, syn_dense_train_y, syn_dense_dev_x, syn_dense_dev_y)
    print('Sparse Synthetic Dev Accuracy for SVM:', svm_dense.svm_accuracy)

    print('\nPerceptron Accuracy For Sparse and Dense Data')

    p_sparse = Classifier('Perceptron', syn_sparse_train_x,
                          syn_sparse_train_y, iterations=10)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if
                    p_sparse.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(syn_sparse_dev_y) * 100
    print('Syn Sparse Dev Accuracy for Perceptron:', accuracy)

    p_dense = Classifier('Perceptron', syn_dense_train_x,
                         syn_dense_train_y, iterations=10)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y))
                    if p_dense.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(syn_dense_dev_y) * 100
    print('Syn Dense Dev Accuracy for Perceptron:', accuracy)

    print('\nAveraged Perceptron Accuracy For Sparse and Dense  Data')

    avg_p_sparse = Classifier('Perceptron', syn_sparse_train_x,
                              syn_sparse_train_y, iterations=10, averaged=True)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if
                    avg_p_sparse.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(syn_sparse_dev_y) * 100

    print('Syn Sparse Dev Accuracy for Averaged Perceptron:', accuracy)

    avg_p_dense = Classifier('Perceptron', syn_dense_train_x,
                             syn_dense_train_y, iterations=10, averaged=True)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if
                    avg_p_dense.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(syn_dense_dev_y) * 100

    print('Syn Dense Dev Accuracy for Averaged Perceptron:', accuracy)

    print('\n Winnow Accuracy For Sparse and Dense Data')

    w_sparse = Classifier('Winnow', syn_sparse_train_x,
                          syn_sparse_train_y, iterations=10)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if
                    w_sparse.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(syn_sparse_dev_y) * 100
    print('Syn Sparse Dev Accuracy for Winnow:', accuracy)

    w_dense = Classifier('Winnow', syn_dense_train_x,
                         syn_dense_train_y, iterations=10)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y))
                    if w_dense.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(syn_dense_dev_y) * 100
    print('Syn Dense Dev Accuracy for Winnow:', accuracy)

    print('\nAveraged Winnow Accuracy For Sparse and Dense Data')
    avg_w_sparse = Classifier(
        'Winnow', syn_sparse_train_x, syn_sparse_train_y, iterations=10, averaged=True)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if
                    avg_w_sparse.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(syn_sparse_dev_y) * 100
    print('Syn Sparse Dev Accuracy for Averaged Winnow:', accuracy)

    avg_w_dense = Classifier('Winnow', syn_dense_train_x,
                             syn_dense_train_y, iterations=10, averaged=True)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if
                    avg_w_dense.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(syn_dense_dev_y) * 100
    print('Syn Dense Dev Accuracy for Averaged Winnow:', accuracy)

    print('\nAdagrad Accuracy For Sparse and Dense Data')

    ada_sparse = Classifier('Adagrad', syn_sparse_train_x,
                            syn_sparse_train_y, iterations=10)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if
                    ada_sparse.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(syn_sparse_dev_y) * 100
    print('Syn Sparse Dev Accuracy for Adagrad:', accuracy)

    ada_dense = Classifier('Adagrad', syn_dense_train_x,
                           syn_dense_train_y, iterations=10)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y))
                    if ada_dense.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(syn_dense_dev_y) * 100
    print('Syn Dense Dev Accuracy for Adagrad:', accuracy)

    print('\nAveraged Adagrad Accuracy For Sparse and Dense Data')

    avg_ada_dense = Classifier(
        'Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, averaged=True)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if
                    avg_ada_dense.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(syn_dense_dev_y) * 100
    print('Syn Dense Dev Accuracy for Averaged Adagrad:', accuracy)

    avg_ada_sparse = Classifier(
        'Adagrad', syn_sparse_train_x, syn_sparse_train_y, iterations=10, averaged=True)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if
                    avg_ada_sparse.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(syn_sparse_dev_y) * 100
    print('Syn Sparse Dev Accuracy for Averaged Adagrad:', accuracy)

# Following functions writes Average Perceptron predictions on synthehic
# data to p-sparse.txt and p.dense.txt
"""
get_predictions_from_syn_test_data(syn_sparse_test_x, "sparse", "Perceptron", avg_p_sparse)
get_predictions_from_syn_test_data(syn_dense_test_x, "dense", "Perceptron", avg_p_dense)

"""
# Following 10 lines write SVM Predictions on Synthetic Data to
# svm-sparse.txt and svm-dense.txt
"""
svm_s = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y)
t = svm_s.get_svm_predictions(syn_sparse_train_x, syn_sparse_train_y, syn_sparse_test_x)
transformed_test_data = t[1]
svm_classifier = t[0]
get_predictions_from_syn_test_data(transformed_test_data, "sparse", "SVM", svm_classifier)
"""
"""
svm_d = Classifier('SVM', syn_dense_train_x, syn_dense_train_y)
t = svm_s.get_svm_predictions(syn_dense_train_x, syn_dense_train_y, syn_dense_test_x)
transformed_test_data = t[1]
svm_classifier = t[0]
get_predictions_from_syn_test_data(transformed_test_data, "dense", "SVM", svm_classifier)
"""

# The function writes Average Perceptron Predictions on CoNLL Data to
# p-connl.txt
"""
get_predictions_from_real_test_data(news_test_x, 'conll', 'Perceptron', avg_p_news)
"""

# The function writes Average Perceptron Predictions on Enron Data to
# p-enron.txt
"""
get_predictions_from_real_test_data(email_test_x, 'enron', 'Perceptron', avg_p_news)
"""
# The following 5 lines write SVM Predictions on CoNLL Data to svm-connl.txt
"""
svm_news = Classifier('SVM', news_train_x, news_train_y)
tup = svm_news.get_svm_predictions(news_train_x, news_train_y,news_test_x)
transformed_test_data = tup[1]
svm_classifier = tup[0]
get_predictions_from_real_test_data(transformed_test_data, 'conll', 'SVM', svm_classifier)
"""
# The following 5 lines write SVM Predictions on Enron Data to svm-enron.txt
"""
svm_news = Classifier('SVM', news_train_x, news_train_y)
tup = svm_news.get_svm_predictions(news_train_x, news_train_y,email_test_x)
transformed_test_data = tup[1]
svm_classifier = tup[0]
get_predictions_from_real_test_data(transformed_test_data, 'enron', 'SVM', svm_classifier)
"""

# Following two functions tune alpha for Winnow and learning rate for Adagrad
"""
tune_alpha_winnow(syn_sparse_train_x, syn_sparse_train_y, syn_sparse_dev_x, syn_sparse_dev_y)
tune_eta_adagrad(syn_sparse_train_x, syn_sparse_train_y, syn_sparse_dev_x, syn_sparse_dev_y)
"""
# Following two functions plot Dev Accuracies of different test sizes for
# 7 algorithms

"""
plot_learning_curves(syn_dense_train_x, syn_dense_train_y, syn_dense_dev_x, syn_dense_dev_y)
plot_learning_curves(syn_sparse_train_x, syn_sparse_train_y, syn_sparse_dev_x, syn_sparse_dev_y)
"""
