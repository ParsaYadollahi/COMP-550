#!/usr/bin/env python
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
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet


NEGFILE = './rt-polaritydata/rt-polarity.neg'
POSFILE = './rt-polaritydata/rt-polarity.pos'
DATAENCODING = "utf-8"
TRAIN_TEST_SPLIT = 0.8
stopwords = set(stopwords.words('english'))
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

    # Get numbers between 0 and len(all data)
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
        word for word in text if word not in stopwords and word not in string.punctuation]

    return basic_preprocess_text


def stem(text):
    text = basic_preprocess(text)
    return " ".join([stemmer.stem(word) for word in text])


def lemmatize(text):
    text = basic_preprocess(text)

    lemmatized_text = []
    for word in text:
        lemmatized_text.append(
            lemmatizer.lemmatize(word, get_wordnet_pos(word)))

    return " ".join(lemmatized_text)


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_train_test(x_train, x_test):
    # Remove all words that occur less than 2 times
    vectorizer = CountVectorizer(
        preprocessor=lemmatize, min_df=4, max_df=0.5)

    # transform text into vector
    train_feature_set = vectorizer.fit_transform(x_train)
    test_feature_Set = vectorizer.transform(x_test)

    # Transform into feature vector
    fv_train = train_feature_set.toarray()
    fv_test = test_feature_Set.toarray()

    return fv_train, fv_test


def lr_Classifier(fv_train, fv_test, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(fv_train, y_train)

    y_pred = lr.predict(fv_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\n Comfusion Matrix: \n", cm)
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


def random_baseline(fv_test, y_test):
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

    # Naïve Bayes
    nb_Classifier_Accuracy = nb_Classifier(
        fv_train, fv_test, y_train, y_test)
    print("Accuracy for NB is: ", nb_Classifier_Accuracy)

    # Logistic Regresssion Algo
    lr_Classifier_Accuracy = lr_Classifier(
        fv_train, fv_test, y_train, y_test)
    print("Accuracy for Logistic Regression is: ", lr_Classifier_Accuracy)

    # Support Vector Machine
    svm_Classifier_Accuracy = svm_Classifier(
        fv_train, fv_test, y_train, y_test)
    print("Accuracy for SVM is: ", svm_Classifier_Accuracy)

    # Random baseline
    rb_Classifier_Accuracy = random_baseline(fv_test, y_test)
    print("Accuracy for Random Baseline: ", rb_Classifier_Accuracy)

    # Random Forest
    rf_Classifier_Accuracy = random_forest_Classifier(
        fv_train, fv_test, y_train, y_test)
    print("Accuracy for random Forest: ", rf_Classifier_Accuracy)


__init__()
