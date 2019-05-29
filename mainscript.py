import pandas as pd
import numpy as np
import os
import re, string, unicodedata
import nltk
import contractions
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters
from collections import defaultdict
from random import sample
from sklearn.model_selection import train_test_split

np.random.seed(500)

# reads one folder
def read_folder(path_to_folder):
    labels, texts = [], []
    for filename in os.listdir(path_to_folder):
        with open(path_to_folder + '/' +filename, "r") as file:
            data = texts.append(file.read())
            label = labels.append(path_to_folder)
    folder_data = pd.DataFrame()
    folder_data['label'] = labels
    folder_data['texst'] = texts
    return folder_data

#reads many folders to a dictionary, key is a label
def read_data(path_to_folder):
    data = pd.DataFrame()
    for foldername in os.listdir(path_to_folder):
        data = pd.concat([data, read_folder(path_to_folder + '/' + foldername)],ignore_index=True)
    return data


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def main():
    #data loading and splitting
    #twenty_training_set = fetch_20newsgroups(subset='train', shuffle=True)
    #twenty_test_set = fetch_20newsgroups(subset='test', shuffle=True)

    #documents = reuters.fileids()
    #reuters_training_set = list(filter(lambda doc: doc.startswith("train"),documents))
    #reuters_test_set = list(filter(lambda doc: doc.startswith("test"),documents))

    bbc_set = read_data("./data/bbc")


main()
