#!/usr/bin/env python
# coding: utf-8

# # Logistic regression

# Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Some of the examples of classification problems are Email spam or not spam, Online transactions Fraud or not Fraud, Tumor Malignant or Benign. Logistic regression transforms its output using the logistic sigmoid function to return a probability value.
# Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability.
# 

# ## What are the types of logistic regression
# 
Binary (eg. Tumor Malignant or Benign)
Multi-linear functions failsClass (eg. Cats, dogs or Sheep's)

# In[42]:


# step 1:import libraries
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[6]:


#step 2 load data file
data = pd.read_csv('titanic.csv')


# In[7]:


data.head()


# In[10]:


print("No of passengers in data:"+str(len(data.index)))


# In[12]:


#step 3: Analysis
sns.countplot(x='Survived',data=data)


# In[13]:


sns.countplot(x='Survived',hue='Sex',data=data)


# In[14]:


sns.countplot(x='Survived',hue='Pclass',data=data)


# In[21]:


data['Age'].plot.hist()


# In[16]:


data.info()


# In[22]:



data['Fare'].plot.hist()


# In[18]:


sns.countplot(x='Survived',hue='Siblings/Spouses Aboard',data=data)


# ## Data wrangling

# In[23]:


data.isnull().sum()


# In[25]:


sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis')


# In[26]:


sns.boxplot(x='Pclass',y='Age',data=data)


# In[27]:


data.head()


# In[29]:


data.dropna(inplace=True)


# In[30]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False)


# In[31]:


sex = pd.get_dummies(data['Sex'],drop_first=True)
pcl = pd.get_dummies(data['Pclass'],drop_first=True)


# In[33]:


data = pd.concat([data,sex,pcl],axis=1)


# In[35]:


data.drop(['Sex','Pclass','Name'],axis=1,inplace=True)


# In[36]:


data.head()


# ## Train data

# In[37]:


x = data.drop('Survived',axis=1)
y = data['Survived']


# In[45]:


from sklearn.model_selection import train_test_split


# In[47]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[48]:


from sklearn.linear_model import LogisticRegression


# In[50]:


logmodel = LogisticRegression()


# In[51]:


logmodel.fit(X_train,y_train)


# In[52]:


prediction = logmodel.predict(X_test)


# In[53]:


from sklearn.metrics import classification_report


# In[54]:


classification_report(y_test,prediction)


# In[55]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)


# In[56]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)


# In[ ]:




