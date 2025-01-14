#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re #strings_search_pattern 
from nltk.corpus import stopwords # dataset/directory of texts of a lang
from nltk.stem.porter import PorterStemmer # primary stemming technique 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split   
from sklearn.linear_model import LogisticRegression #binary classification technique 
from sklearn.metrics import accuracy_score #quantitfying the perfomance 


# In[4]:


import nltk 
nltk.download('stopwords')

print(stopwords.words('english')
# In[7]:


ds = pd.read_csv("C:\\Users\\shrey\\Desktop\\programs\\datasets\\datasets\\fake_news_train.csv") #data_preprocessing 


# In[9]:


ds.shape


# In[10]:


print(ds.head())


# In[11]:


ds.isnull().sum()


# In[12]:


ds=ds.fillna("")


# In[16]:


ds['content'] = ds['title']+" "+ds['author']
print(ds['content'])


# In[18]:


x = ds.drop(columns='label',axis=1)
y = ds["label"]
print(x)
print(y)


# In[19]:


port_stem = PorterStemmer()


# In[71]:


def stemming(content):
 stem_cont = re.sub('[a-zA-Z]',' ',content) #characters from a to z will get replaced by blank space 
  #re is used for specifying for a string that matches   
 stem_cont = stem_cont.lower() #lower case for consistency
    
 stem_cont = stem_cont.split() #tokenization
    
 stem_cont = [port_stem.stem(words) for words in stem_cont if not words in stopwords.words('english')]
              
 stem_cont = ' '.join(stem_cont)
    
 return stem_cont


# In[70]:


ds['content'] = ds['content'].apply(stemming)


# In[37]:


x = ds["content"].values 
y = ds["label"].values 


# In[48]:


x 


# In[49]:


x.shape


# In[40]:


y 


# In[41]:


y.shape


# In[54]:


vectorizer = TfidfVectorizer()
vectorizer.fit(x)

x = vectorizer.transform(x)


# In[55]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, stratify = y, random_state = 2)


# In[58]:


model = LogisticRegression()
model.fit(x_train,y_train)


# In[60]:


x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy score of the training data:", training_data_accuracy)


# In[61]:


x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy score of the training data:", testing_data_accuracy)


# In[67]:


#making_a_predictive_system 
x_new = x_test[3]
prediction = model.predict(x_new)
print(prediction)
if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[64]:


print(y_test[3])


# In[ ]:





# In[ ]:




