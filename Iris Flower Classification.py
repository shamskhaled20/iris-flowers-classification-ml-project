#!/usr/bin/env python
# coding: utf-8

# Loading the dataset:

# In[24]:


# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[25]:


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'


# In[26]:


col_name = ['sepal-lenght','sepal-width','petal-lenght','petal-width','class']


# In[27]:


df = pd.read_csv(url, names = col_name)


# In[28]:


df.head()


# In[29]:


df.describe()


# In[30]:


df.shape


# In[31]:


df.info


# In[32]:


df['class'].value_counts()


# Data Visualization

# In[33]:


sns.violinplot(y='class', x='sepal-lenght', data=df, inner='quartile')
plt.show()
sns.violinplot(y='class', x='sepal-width', data=df, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal-lenght', data=df, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal-width', data=df, inner='quartile')
plt.show()


# In[34]:


sns.pairplot(df, hue='class', markers='+')
plt.show()


# In[35]:


plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r')
plt.show()


# In[36]:


x = df.drop(['class'], axis=1)
y = df['class']
print(f'X shape: {x.shape} | y shape: {y.shape} ')


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)


# In[43]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))
# evaluate each model in turn
results = []
model_names = []
for name, model in models:
  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  model_names.append(name)
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[45]:


model = SVC(gamma='auto')
model.fit(x_train, y_train)
prediction = model.predict(x_test)


# In[46]:


print(f'Test Accuracy: {accuracy_score(y_test, prediction)}')
print(f'Classification Report: \n {classification_report(y_test, prediction)}')


# In[ ]:




