import pandas as pd
import numpy as np
import os
import re, string, unicodedata
import nltk
import contractions
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters

np.random.seed(500)

def read_folder(path_to_folder):
    files = {}
    for filename in os.listdir(path_to_folder):
        with open(path_to_folder + '/' +filename, "r") as file:
            if filename in files:
                continue
            files[filename] = file.read()
    return files

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
    data = read_folder("./data/bbc/business")

main()
