import os
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
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

TEXT_PATH = "/home/klesisz/Pulpit/bbc/"


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


def main():
    bbc_data = read_data(TEXT_PATH)
    print("Preprocessing data")
    preprocess(bbc_data)
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

    vectorized_train_documents = vectorizer.fit_transform(bbc_training_set_x)
    vectorized_test_documents = vectorizer.transform(bbc_test_set_x)

    # OVO
    print("\n--------------\nOVO")
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, tol=1e-5)
    SVM.fit(vectorized_train_documents, bbc_training_set_y)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(vectorized_test_documents)

    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, bbc_test_set_y) * 100)

    # OVA

    print("\n--------------\nOVA")

    # Classifier - Algorithm - SVC
    # fit the training dataset on the classifier
    SVM = svm.LinearSVC(C=1.0, tol=1e-5)
    SVM.fit(vectorized_train_documents, bbc_training_set_y)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(vectorized_test_documents)

    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, bbc_test_set_y) * 100)
    print("--------------")


if __name__ == '__main__':
    main()
