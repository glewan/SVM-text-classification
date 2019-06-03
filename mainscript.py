#-*- coding: utf-8 -*-
import os
import re

import nltk
import gensim
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


np.random.seed(500)  # in order to have the exact same result for each run
nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = stopwords.words("english")  # stop words cache

TEXT_PATH = u'C:\\Users\\a\\Desktop\\WedtDane\\bbc'


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

def FrameTNG (set):
    # creating dataframe
    df = pd.DataFrame([set.target.tolist(), set.data]).T
    df.columns = ['label1', 'text']

    targets = pd.DataFrame(set.target_names)
    targets.columns = ['label']

    df = pd.merge(df, targets, left_on='label1', right_index=True)
    df = df[['label', 'text']]
    return df


def main():
    #bbc_data = read_data(TEXT_PATH)

    from nltk.corpus import reuters


    from sklearn.datasets import fetch_20newsgroups
    # Loading training set and test set; removing headers and footers
    #cats = ['alt.atheism', 'sci.space']
    #TNG_training_set = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'))
    #TNG_test_set = fetch_20newsgroups(subset='test', categories=cats, remove=('headers', 'footers', 'quotes'))
    TNG_training_set = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    TNG_test_set = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    TNG_train_data = FrameTNG(TNG_training_set)
    TNG_test_data = FrameTNG(TNG_test_set)

    print("Preprocessing data")
    preprocess(TNG_train_data)
    preprocess(TNG_test_data)

    print(TNG_train_data)
    print(TNG_test_data)


    #(bbc_data)
    #bbc_training_set_x, bbc_test_set_x, bbc_training_set_y, bbc_test_set_y = split_data(bbc_data['final'],
    #                                                                                    bbc_data['label'], 0.3)

    TNG_training_set_x = TNG_train_data.loc[:,'final']
    TNG_training_set_y = TNG_train_data.loc[:, 'label']
    TNG_test_set_x = TNG_test_data.loc[:,'final']
    TNG_test_set_y = TNG_test_data.loc[:,'label']

    print(TNG_training_set_x)
    print(TNG_training_set_y)
    print(TNG_test_set_x)
    print(TNG_test_set_y)

    # label encode the target variable
    Encoder = LabelEncoder()
    #bbc_training_set_y = Encoder.fit_transform(bbc_training_set_y)
    #bbc_test_set_y = Encoder.fit_transform(bbc_test_set_y)

    #print("bbc print x 1")
    #print(bbc_training_set_x)
    #print("bbc print x 2")
    #print(bbc_test_set_x)

    #print("bbc print y 1")
    #print(bbc_training_set_y)
    #print("bbc print y 2")
    #print(bbc_test_set_y)


    TNG_training_set_y = Encoder.fit_transform(TNG_training_set_y)
    TNG_test_set_y = Encoder.fit_transform(TNG_test_set_y)

    print("tng print x 1")
    print(TNG_training_set_x)
    print("tng print x 2")
    print(TNG_test_set_x)

    print("tng print y 1")
    print(TNG_training_set_y)
    print("tng print y 2")
    print(TNG_test_set_y)


    vectorizer = TfidfVectorizer(min_df=3,
                                 max_df=0.90,
                                 max_features=3000,
                                 use_idf=True,
                                 sublinear_tf=True,
                                 norm='l2')

    #vectorized_train_documents = vectorizer.fit_transform(bbc_training_set_x)
    #vectorized_test_documents = vectorizer.transform(bbc_test_set_x)

    vectorized_train_documents_TNG = vectorizer.fit_transform(TNG_training_set_x)
    vectorized_test_documents_TNG = vectorizer.transform(TNG_test_set_x)

    # OVO
    print("\n--------------\nOVO")
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, tol=1e-5)
    SVM.fit(vectorized_train_documents_TNG, TNG_training_set_y)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(vectorized_test_documents_TNG)

    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, TNG_test_set_y) * 100)

    # OVA

    print("\n--------------\nOVA")

    # Classifier - Algorithm - SVC
    # fit the training dataset on the classifier
    SVM = svm.LinearSVC(C=1.0, tol=1e-5)
    SVM.fit(vectorized_train_documents_TNG, TNG_training_set_y)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(vectorized_test_documents_TNG)

    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, TNG_test_set_y) * 100)
    print("--------------")


if __name__ == '__main__':
    main()
