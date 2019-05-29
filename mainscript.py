import pandas as pd
import numpy as np
import os
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(500)
stop_words = stopwords.words("english")

# reads one folder
def read_folder(path_to_folder):
    labels, texts = [], []
    for filename in os.listdir(path_to_folder):
        with open(path_to_folder + '/' +filename, "r") as file:
            data = texts.append(file.read())
            label = labels.append(path_to_folder)
    folder_data = pd.DataFrame()
    folder_data['label'] = labels
    folder_data['text'] = texts
    return folder_data

#reads many folders to a dataFrame
def read_data(path_to_folder):
    data = pd.DataFrame()
    for foldername in os.listdir(path_to_folder):
        data = pd.concat([data, read_folder(path_to_folder + '/' + foldername)], ignore_index=True)
    return data

def split_data(data, split_ratio):
    training_set = pd.DataFrame()
    test_set = pd.DataFrame()
    data_training, data_test = train_test_split(data, test_size= split_ratio, random_state=42)
    return data_training, data_test

def preprocessing(data):
    #tokenization,
    for index, row in data.iterrows():
        data.at[index, 'text'] = tokenize(data.at[index, 'text'])
        #representer = tf_idf(data['text'])


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in stop_words]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens =list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens))
    return filtered_tokens

def tf_idf(text):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2')
    tfidf.fit(text)
    return tfidf


def main():
    #data loading and splitting
    #twenty_training_set = fetch_20newsgroups(subset='train', shuffle=True)
    #twenty_test_set = fetch_20newsgroups(subset='test', shuffle=True)

    #reuters_data = reuters.fileids()
    #reuters_training_set = list(filter(lambda doc: doc.startswith("train"),reuters_data))
    #reuters_test_set = list(filter(lambda doc: doc.startswith("test"),reuters_data))

    bbc_data = read_data("./data/bbc")
    bbc_training_set, bbc_test_set = split_data(bbc_data, 0.3)
    #data_preprocessing
    preprocessing(bbc_training_set)
    preprocessing(bbc_test_set)

    bbc_data = read_data("./data/bbc")

main()
