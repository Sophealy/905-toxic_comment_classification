#!/usr/bin/env python
# coding: utf-8

import click
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

spell = SpellChecker()
lemmatizer = WordNetLemmatizer()




def preprocessing1(comment):
    """
    PREPROCESSING 1
    - Punctuation removal
    - Tokenization by whitespace
    - Stop words
    - Number removal
    - Shortword removal
    - Lemmatization
    """

    # Punctutation removal
    comment = ''.join([char for char in comment if char not in string.punctuation])

    # Tokenization and lowercase
    comment_tok = comment.lower().split(' ')

    # Stopwords and short word
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in comment_tok if token not in stop_words and len(token) >1 ]

    # lemmatization
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(lemmas)


def preprocessing2(comment):
    """
    PREPROCESSING 2
    - English tokenization
    - Mispelling correction
    - Lemmatization
    """
    comment_toks = word_tokenize(comment)
    comment_toks = [spell.correction(tok) for tok in comment_toks]

    # Lemmatization
    lemmas = [lemmatizer.lemmatize(t) for t in comment_toks]

    return ' '.join(lemmas)



def preprocessing(comment,number):
    if number == 1:
        return preprocessing1(comment)
    elif number == 2:
        return preprocessing2(comment)
    else:
        print("Pipeline not defined")


def read_csv(filepath,verbose,sep=","):
    df = pd.read_csv(filepath,sep=sep)
    if verbose:
        click.echo("{} was read.".format(filepath))
    return df

def write_csv(df,out_path,verbose):
    df.to_csv(out_path,index=None)
    if verbose:
        click.echo("The dataframe is now export into a csv file called {} ".format(out_path))




@click.command()
@click.option('--verbose', is_flag=True, help="Will print verbose messages.")
@click.argument('csv_in')
@click.argument('csv_out')
@click.option('--number', default=2, help="Preprocessing pipeline (1 or 2)")
def main(verbose , csv_in ,csv_out,number):
    """ Preprocess the csv file and return the current csv with new features."""

    # Read the csv
    df = read_csv(csv_in,verbose)

    # preprocessing
    df['comment_text'] = df['comment_text'].apply(lambda x: preprocessing(x,number))

    # Export the csv
    write_csv(df,csv_out,verbose)

if __name__ == "__main__":
    main()
