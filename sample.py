#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[2]:


df_train = pd.read_csv('titanic_data.csv')
df_test = pd.read_csv('tested.csv')


# In[3]:


print("Rows of training dataset :", len(df_train))
print("Rows of testing dataset :", len(df_test))


# In[4]:


df_train.head()


# In[5]:


Survivors = df_train.Survived.sum()
print("Survived :", Survivors)
print("Died :", len(df_train)- Survivors)


# In[6]:


print(df_train.isnull().sum())


# In[7]:


print(df_test.isnull().sum())


# In[8]:


df_train[['Pclass','Survived']].groupby('Pclass').mean().sort_values(by='Survived', ascending = False).reset_index()


# Upper class people are more likely to survive.

# In[9]:


plt.hist(df_train['Fare'],bins = 20,edgecolor = 'white', linewidth = 2)
plt.show()


# # Feature Engineering

# In[10]:


df_train.drop('PassengerId', axis = 1, inplace = True)
df_test.drop('PassengerId', axis = 1, inplace = True)


# In[11]:


mapping_class = {1:1, 2:0, 3:-1}
df_train['Pclass'] = df_train['Pclass'].map(mapping_class)
df_test['Pclass'] = df_test['Pclass'].map(mapping_class)


# In[12]:


df_train.drop('Name', axis = 1, inplace = True)
df_test.drop('Name', axis = 1, inplace = True)


# In[13]:


df_train.loc[df_train.Age.isnull(),'Age'] = df_train['Age'].median()
df_test.loc[df_test.Age.isnull(),'Age'] = df_test['Age'].median()


# In[14]:


df_train['Family'] = df_train['Parch'] + df_train['SibSp'] + 1
df_test['Family'] = df_test['Parch'] + df_test['SibSp'] + 1


# In[15]:


df_train.head(10)


# In[16]:


df_test.head(10)


# In[17]:


df_train[['Family','Survived']].groupby('Family').mean().sort_values('Survived', ascending = False).reset_index()


# In[18]:


df_train['FamilySize'] = 'Big'
df_train.loc[df_train['Family']<=4,'FamilySize'] = 'Small'
df_train.loc[df_train['Family'] == 1, 'FamilySize'] = 'Single'

df_test['FamilySize'] = 'Big'
df_test.loc[df_test['Family']<4,'FamilySize'] = 'Small'
df_test.loc[df_test['Family'] == 1, 'FamilySize'] = 'Single'


# In[19]:


df_train.drop(['SibSp', 'Parch', 'Family'], axis = 1, inplace = True)
df_test.drop(['SibSp', 'Parch', 'Family'], axis = 1, inplace = True)


# In[20]:


df_train.head()


# In[21]:


df_test.drop('Ticket', axis = 1, inplace = True)


# In[22]:


df_test.drop('Cabin', axis =1, inplace = True)


# In[23]:


df_train.drop(['Ticket', 'Cabin'], axis = 1, inplace = True)


# In[24]:


df_train.head()


# In[25]:


df_test['Embarked'].unique()


# In[26]:


df_train.loc[df_train['Embarked'].isnull()]


# In[27]:


df_train[['Embarked', 'Survived']].groupby('Embarked').mean().sort_values('Survived').reset_index()


# In[28]:


df_train.loc[df_train['Embarked'].isnull()]


# In[29]:


df_train.drop([61,829], axis = 0, inplace = True)


# In[30]:


df_train.isnull().sum()


# In[31]:


df_train.head()


# In[32]:


df_test.isnull().sum()


# In[33]:


df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()


# In[34]:


df_test.isnull().sum()


# In[35]:


df_test.head()


# In[36]:


df_train = pd.get_dummies(df_train,prefix = 'ship',drop_first=True)
df_test = pd.get_dummies(df_test,prefix = 'ship',drop_first=True)


# In[37]:


df_train.head()


# In[38]:


Y_train = df_train['Survived']
X_train = df_train.drop('Survived', axis = 1)

X_test = df_test.drop('Survived', axis = 1)


# In[39]:


model = LogisticRegression(random_state = 0, max_iter = 200)
model.fit(X_train, Y_train)


# In[40]:


Y_test = model.predict(X_test)


# In[41]:


Y_test = pd.DataFrame(Y_test)


# In[42]:


Y_test.head()


# In[43]:


df_test['Survived'].head()


# In[44]:


score = model.score(X_test, Y_test)
print(score)


# In[49]:


def get_matrix(X,Y):
    cm = metrics.confusion_matrix(X,Y)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


# In[50]:


get_matrix(df_test['Survived'], Y_test)


# In[ ]:




