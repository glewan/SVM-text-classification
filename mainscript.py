#-*- coding: utf-8 -*-
import os
import re
import sys
import nltk
import string
import time
import gensim
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix

np.random.seed(500)  # in order to have the exact same result for each run
nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = stopwords.words("english")  # stop words cache


TEXT_PATH = u'C:\\Users\\a\\Desktop\\WedtDane\\bbc'
#TEXT_PATH = u'C:\\Users\\Gabrysia\\Desktop\\STUDIA\\WEDT\\projekt\\data\\bbc'

# reads one folder
def read_folder(path_to_folder):
    print("Starting import from: " + path_to_folder)
    labels, texts = [], []
    for filename in os.listdir(path_to_folder):
        with open(os.path.join(path_to_folder, filename), "r", encoding="ISO-8859-1") as file:
            texts.append(file.read())
            labels.append(os.path.split(path_to_folder)[1])
    folder_data = pd.DataFrame()
    folder_data['label'] = labels
    folder_data['text'] = texts
    return folder_data


# reads many folders to a dataFrame
def read_data(path_to_folder):
    print("Starting data read")
    data = pd.DataFrame()
    for foldername in os.listdir(path_to_folder):
        data = pd.concat([data, read_folder(os.path.join(path_to_folder, foldername))], ignore_index=True)

    print(data)
    print("Finished reading data")
    return data


def split_data(text, labels, split_ratio):
    """
    :param data:
    :param split_ratio:
    :return: train_x: training data predictors
    :return: train_y: training data target
    :return: test_x: test data predictors
    :return: test_y: test data target
    """
    train_x, test_x, train_y, test_y = train_test_split(text, labels, test_size=split_ratio, random_state=42)
    return train_x, test_x, train_y, test_y


def preprocess(data):
    text = data['text']
    min_length = 3
    for index, entry in enumerate(text):
        # change text to lowercase, split into individual words
        words = [word.lower() for word in nltk.word_tokenize(entry)]
        # remove stopwords
        words = [word for word in words if word not in STOP_WORDS]
        # word stemming/lemmatization
        tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
        # filter out non-alphabetic words and tokens shorter that min_length
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
        data.loc[index, 'final'] = str(filtered_tokens)


def frameNLTK_set(nltk_set):
    documents = nltk_set.fileids()

    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                               documents))

    train_docs = [nltk_set.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [nltk_set.raw(doc_id) for doc_id in test_docs_id]
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([nltk_set.categories(doc_id)
                                      for doc_id in train_docs_id])#).tolist()
    test_labels = mlb.transform([nltk_set.categories(doc_id)
                                 for doc_id in test_docs_id])#).tolist()

    training_set = pd.DataFrame()
    training_set['text'] = train_docs
    test_set = pd.DataFrame()
    test_set['text'] = test_docs

    return training_set, train_labels, test_set, test_labels


def bbc():
    bbc_data = read_data(TEXT_PATH)

    print("Preprocessing data")
    start = time.time()
    preprocess(bbc_data)
    end = time.time()
    print("Executed within:", end - start)

    bbc_training_set_x, bbc_test_set_x, bbc_training_set_y, bbc_test_set_y = split_data(bbc_data['final'],
                                                                                       bbc_data['label'], 0.3)

    # label encode the target variable
    Encoder = LabelEncoder()

    bbc_training_set_y = Encoder.fit_transform(bbc_training_set_y)
    bbc_test_set_y = Encoder.fit_transform(bbc_test_set_y)

    vectorizer = TfidfVectorizer(min_df=3,
                                 max_df=0.90,
                                 max_features=3000,
                                 use_idf=True,
                                 sublinear_tf=True,
                                 norm='l2')

    vectorized_train_documents_bbc = vectorizer.fit_transform(bbc_training_set_x)
    vectorized_test_documents_bbc = vectorizer.transform(bbc_test_set_x)

    return vectorized_train_documents_bbc, bbc_training_set_y, vectorized_test_documents_bbc, bbc_test_set_y

def reut():
    print("Importing Reuters-21578 dataset")
    from nltk.corpus import reuters

    reuters_training_set, reuters_training_set_y, reuters_test_set, reuters_test_set_y = frameNLTK_set(reuters)
    print("Preprocessing data")
    start = time.time()
    preprocess(reuters_training_set)
    preprocess(reuters_test_set)
    end = time.time()
    print("Executed within:",end - start)

    reuters_training_set_x = reuters_training_set.loc[:, 'final']
    reuters_test_set_x = reuters_test_set.loc[:, 'final']

    vectorizer = TfidfVectorizer(min_df=3,
                                 max_df=0.90,
                                 max_features=3000,
                                 use_idf=True,
                                 sublinear_tf=True,
                                 norm='l2')
    vectorized_train_documents_reuters = vectorizer.fit_transform(reuters_training_set_x)
    vectorized_test_documents_reuters = vectorizer.transform(reuters_test_set_x)

    return vectorized_train_documents_reuters, reuters_training_set_y, vectorized_test_documents_reuters, reuters_test_set_y

def preprocessTNG(set):
    X, labels = [], []

    for i, entry in enumerate(set['data']):
        tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(entry)]
        text = [f.lower() for f in tokens if f and f.lower() not in STOP_WORDS]
        if (len(text) == 0):
            continue
        Index = set['target'][i]
        X.append(text)
        labels.append(Index)
    return X, np.array(labels)


def tng():
    print("Importing 20 News Group dataset")
    #cats = ['alt.atheism', 'sci.space']
    #TNG_set = fetch_20newsgroups(subset='all', categories=cats, remove=('headers', 'footers', 'quotes'),
      #                               shuffle=True, random_state=42)

    TNG_set = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

    print("Preprocessing data")
    start = time.time()
    Data, Labels = preprocessTNG(TNG_set)
    end = time.time()
    print("Executed within:", end - start)

    # rows: Docs. columns: words
    Data = np.array([np.array(xi) for xi in Data])

    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(Data)
    vectorized_data = vectorizer.transform(Data)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1).split(vectorized_data, Labels)
    train_indices, test_indices = next(sss)

    train_x, test_x = vectorized_data[train_indices], vectorized_data[test_indices]
    train_y, test_y = Labels[train_indices], Labels[test_indices]

    return train_x, train_y, test_x, test_y

def evaluate(test, predictions):
    precision = precision_score(test, predictions,
                                average='micro')
    recall = recall_score(test, predictions,
                          average='micro')
    f1 = f1_score(test, predictions, average='micro')

    print("Micro-average quality numbers: \n")
    print("Precision: {:.4f},\nRecall: {:.4f},\nF1-measure: {:.4f}\n"
          .format(precision, recall, f1))

    precision = precision_score(test, predictions,
                                average='macro')
    recall = recall_score(test, predictions,
                          average='macro')
    f1 = f1_score(test, predictions, average='macro')

    print("Macro-average quality numbers:\n")
    print("Precision: {:.4f},\nRecall: {:.4f},\nF1-measure: {:.4f}\n"
          .format(precision, recall, f1))

    print("--------------")

def print_confm(test, predictions, model):
    if (model == 'b') or (model == 'g'):
        confm = confusion_matrix(test, predictions)
        print("Confusion Matrix:\n")
        print(confm)
    elif (model == 'r'):
        confm = multilabel_confusion_matrix(test, predictions)
        print("Confusion Matrix:\n")
        print(confm)

def main():

    print("Welcome to SVM text classifier. Please choose a dataset: \n"
          "Press 'b' for BBC news dataset\n"
          "Press 'r' for Reuters-21578 dataset\n"
          "Press 'g' for 20 News group\n"
          "Press 'q' for exit\n \n"
          "Enter your decision: ")

    model = input()

    if model == 'b':
        train_X, train_Y, test_X, test_Y = bbc()
    elif model == 'r':
        train_X, train_Y, test_X, test_Y = reut()
    elif model == 'g':
        train_X, train_Y, test_X, test_Y = tng()
    elif model == 'q':
        print("Program is closing...")
        sys.exit(0)
    else:
        "Please choose one of described options"

    # OVO
    print("\n--------------\nOVO")
    if (model == 'b') or (model == 'g'):
        classifier = OneVsOneClassifier(LinearSVC(random_state=42))
        classifier.fit(train_X, train_Y)

        predictions_SVM = classifier.predict(test_X)
        evaluate(test_Y, predictions_SVM)
        print_confm(test_Y, predictions_SVM, model)

        # OVA
        print("\n--------------\nOVA")
        classifier = OneVsRestClassifier(LinearSVC(random_state=42))
        classifier.fit(train_X, train_Y)

        predictions_SVM = classifier.predict(test_X)

        evaluate(test_Y, predictions_SVM)
        print_confm(test_Y, predictions_SVM, model)

    if (model == 'r'):
        # OVA
        print("\n--------------\nOVA")
        # classifier = OneVsRestClassifier(LinearSVC(random_state=42))

        classifier = ClassifierChain(
            classifier=LinearSVC(),
            require_dense=[False, True]
        )

        classifier.fit(train_X, train_Y)
        predictions_SVM = classifier.predict(test_X)
        evaluate(test_Y, predictions_SVM)
        print_confm(test_Y, predictions_SVM, model)

if __name__ == '__main__':
    main()
