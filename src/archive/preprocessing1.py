#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("train.csv")
print(df.shape)


# In[3]:


print(df.dtypes)


# In[4]:


comment = df['comment_text']
print(comment.head())
comment = comment.as_matrix()


# In[5]:


label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
print(label.head())
label = label.as_matrix()


# In[6]:


#ct1 counts samples having atleast one label
#ct2 counts samples having 2 or more than 2 labels

ct1,ct2 = 0,0
for i in range(label.shape[0]):
    ct = np.count_nonzero(label[i])
    if ct :
        ct1 = ct1+1
    if ct>1 :
        ct2 = ct2+1
print(ct1)
print(ct2)


# In[7]:


comments = []
labels = []

for ix in range(comment.shape[0]):
    if len(comment[ix])<=400:
        comments.append(comment[ix])
        labels.append(label[ix])


# In[8]:


labels = np.asarray(labels)


# In[9]:


print(len(comments))


# In[10]:


# remove punctuation
import string
print(string.punctuation)
comments = [''.join(c for c in s if c not in string.punctuation) for s in comments]


# In[11]:


comments


# In[12]:


#spliting the comments into individual words
comments = [word for line in comments for word in line.split()]


# In[13]:


# remove stopwords
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))  
tokens = [token for token in comments if token not in stop_words]
raw = tokens


# In[15]:


import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
outtab = "                                         "
trantab = str.maketrans(punctuation_edit, outtab)
#create objects for stemmer and lemmatizer
lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()
#download words from wordnet library
# nltk.download('wordnet')
for i in range(len(comments)):
    comments[i] = comments[i].lower().translate(trantab)
    l = []
    for word in comments[i]:
        l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
    comments[i] = "".join(l)


# In[17]:


pos = nltk.pos_tag(comments)


# In[22]:


from pandas import DataFrame
output = {'Preprocess' : comments,
         'POS' : pos}
df = DataFrame(output, columns= ['Preprocess', 'POS'])

export_csv = df.to_csv(r"export_dataframe.csv", index = None, header=True)
print(df)


# In[ ]:




