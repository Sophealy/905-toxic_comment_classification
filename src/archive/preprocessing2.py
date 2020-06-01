#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer


# In[2]:


train_text = pd.read_csv("train.csv")
print(train_text.dtypes)


# In[3]:


comment = train_text['comment_text']
print(comment.head())
comment = comment.as_matrix()


# In[4]:


label = train_text[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
print(label.head())
label = label.as_matrix()


# In[5]:


comments = []
labels = []

for ix in range(comment.shape[0]):
    if len(comment[ix])<=400:
        comments.append(comment[ix])
        labels.append(label[ix])


# In[50]:


# remove punctuation
import string
print(string.punctuation)
comments = [''.join(c for c in s if c not in string.punctuation) for s in comments]


# In[36]:


# split tokens by white space
comments = [word for line in comments for word in line.split()]


# In[37]:


# remove tokens that are not alphabetic
comments = [comment for comment in comments if comment.isalpha()]


# In[38]:


# convert letters to lower case
comments = [comment.lower() for comment in comments]


# In[39]:


# remove stopwords
stop_words = set(stopwords.words('english'))
comments = [comment for comment in comments if comment not in stop_words]


# In[40]:


# remove short words (one letter)
comments = [comment for comment in comments if len(comment) > 1]


# In[42]:


# lemmatization
lem = WordNetLemmatizer()
comments = [lem.lemmatize(comment,"v") for comment in comments]
sentence = ' '.join(comments)


# In[53]:


import nltk
pos = nltk.pos_tag(comments)


# In[54]:


from pandas import DataFrame
output = {'Preprocess' : comments,
         'POS' : pos}
df = DataFrame(output, columns= ['Preprocess', 'POS'])

export_csv = df.to_csv(r"export_dataframe.csv", index = None, header=True)
print(df)


# In[ ]:




