#!/usr/bin/env python
# coding: utf-8

# # GBDT

#  It seems that multi regression is not supported natively, see [StackOverFlow](https://stackoverflow.com/questions/21556623/regression-with-multi-dimensional-targets)

# ## Importing packages

# In[62]:


import numpy as np
from sklearn.ensemble import  GradientBoostingRegressor


import pandas as pd


# In[63]:


x_train = np.random.randn(30,10)
y_train = np.random.randn(30,1)



x_test = np.random.randn(30,10)
y_test = np.random.randn(30,1)

y_train


# In[64]:


gbc = GradientBoostingRegressor()
gbc.fit(x_train, y_train)


# In[ ]:





# In[ ]:




