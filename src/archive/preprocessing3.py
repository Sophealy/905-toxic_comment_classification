
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer
import csv
import re

def pre_process(text_record):

    # split tokens by white space
    tokens = text_record.split()
    # convert letters to lower case
    tokens = [token.lower() for token in tokens]
    #word tokenization
    word_tokens = nltk.word_tokenize(tokens)
    word_tokens = nltk.pos_tag(word_tokens)

    return word_tokens




with open('train.csv') as csvfile:
    train_text_clean = [pre_process(comment) for comment in csvfile]
    print(train_text_clean)
    # train_text_clean = [s for s in train_text_clean if s]
    # writer = csv.writer(open("output.csv", 'w'))
    # writer.writerow(train_text_clean)