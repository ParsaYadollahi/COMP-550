#!/usr/bin/env python
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import random

# NUMPY
import numpy as np

# SKLEARN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# NLTK

NEGFILE = './rt-polaritydata/rt-polarity.neg'
POSFILE = './rt-polaritydata/rt-polarity.pos'
DATAENCODING = "utf-8"
TRAIN_TEST_SPLIT = 0.8  # define ratio of dataset used for training vs testing
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def getData():
    pos_data = []
    neg_data = []

    with open(POSFILE, 'r', encoding=DATAENCODING, errors='ignore') as fr:
        pos_data = list(line.rstrip() for line in fr.readlines())

    with open(NEGFILE, 'r', encoding=DATAENCODING, errors='ignore') as fr:
        neg_data = list(line.rstrip() for line in fr.readlines())

    # Split the dataset into two - Data and Training set
    # Replace (Positive = 1 | negative = 0)
    split_data = [[line, 1] for line in pos_data] + [[line, 0]
                                                     for line in neg_data]
    N = len(split_data)
    perms = np.random.permutation(N)

    # Get numners between 0 and len(all data)
    N_train = int(TRAIN_TEST_SPLIT * N)

    x_train, y_train = [split_data[i][0] for i in perms[:N_train]], [
        split_data[i][1] for i in perms[:N_train]]
    x_test, y_test = [split_data[i][0] for i in perms[N_train:]], [
        split_data[i][1] for i in perms[N_train:]]

    return ([x_train, y_train], [x_test, y_test])


def basic_preprocess(text):
    # Lower case all characters
    text = text.lower()

    text = word_tokenize(text)

    # Remove stopwords and punctuation
    basic_preprocess_text = [
        word for word in text if word not in STOPWORDS and word not in string.punctuation]

    return basic_preprocess_text


def stemming(text):
    text = basic_preprocess(text)
    return " ".join([stemmer.stem(word) for word in text])


def lemmatization(text):
    text = basic_preprocess(text)
    return " ".join([lemmatizer.lemmatize(word) for word in text])


def preprocess_train_test(x_train, x_test):
    # Remove all words that occur less than 2 times
    vectorizer = CountVectorizer(preprocessor=stemming, min_df=2, max_df=0.8)

    # transform text into vector
    train_feature_set = vectorizer.fit_transform(x_train)
    test_feature_Set = vectorizer.transform(x_test)

    # Transform into feature vector
    fv_train = train_feature_set.toarray()
    fv_test = test_feature_Set.toarray()

    return fv_train, fv_test


# class Logistic_Regression_Classifier:
#     def __init__(self):
#       # I just spent the past 3 hours trying to do linear regression
#       # When I was supposed to be doing logistic :')
#         regressor = LogisticRegression(random_state=0, C=2)
#         self.regressor = regressor

#     def fit(self, x_train, y_train):
#         self.x_train = x_train
#         self.y_train = y_train


#         self.regressor.fit(x_train, y_train)
#         return self

#     def predict(self, x_test):
#         y_pred = self.regressor.predict(x_test)
#         return y_pred


def lr_Classifier(fv_train, fv_test, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(fv_train, y_train)

    y_pred = lr.predict(fv_test)
    return calculate_accuracy(y_test, y_pred)


def svm_Classifier(fv_train, fv_test, y_train, y_test):
    clf = SVC(kernel='linear')
    clf.fit(fv_train, y_train)
    y_pred = clf.predict(fv_test)
    return calculate_accuracy(y_test, y_pred)


def nb_Classifier(fv_train, fv_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(fv_train, y_train)
    y_pred = gnb.predict(fv_test)
    return calculate_accuracy(y_test, y_pred)


def random_baseline_Classifier(fv_test, y_test):
    y_pred = []
    for i in range(len(fv_test)):
        y_pred.append(random.randint(0, 1))
    return calculate_accuracy(y_test, y_pred)


def random_forest_Classifier(fv_train, fv_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(fv_train, y_train)
    y_pred = clf.predict(fv_test)
    return calculate_accuracy(y_test, y_pred)


def calculate_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def __init__():
    # Get training and testing data
    training_data, testing_data = getData()

    # Split feature vector and features
    x_train, y_train = training_data
    x_test, y_test = testing_data

    # Feature vectors
    fv_train, fv_test = preprocess_train_test(x_train, x_test)

    # Logistic Regresssion Algo
    lr_Classifier_Accuracy = lr_Classifier(
        fv_train, fv_test, y_train, y_test)

    print("Accuracy for Logistic Regression is: ", lr_Classifier_Accuracy)

    # Support Vector Machine
    svm_Classifier_Accuracy = svm_Classifier(
        fv_train, fv_test, y_train, y_test)

    print("Accuracy for SVM is: ", svm_Classifier_Accuracy)

    # Naïve Bayes
    nb_Classifier_Accuracy = nb_Classifier(
        fv_train, fv_test, y_train, y_test)

    print("Accuracy for NB is: ", nb_Classifier_Accuracy)

    # Random baseline
    rb_Classifier_Accuracy = random_baseline_Classifier(fv_test, y_test)
    print("Accuracy for Random Baseline: ", rb_Classifier_Accuracy)

    # Random Forest
    rf_Classifier_Accuracy = random_forest_Classifier(
        fv_train, fv_test, y_train, y_test)

    print("Accuracy for random Forest: ", rf_Classifier_Accuracy)


__init__()
