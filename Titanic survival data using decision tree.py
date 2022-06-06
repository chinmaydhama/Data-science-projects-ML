#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'


# In[5]:


titanic = pd.read_csv(url)


# In[7]:


df=titanic


# In[8]:


df.head()


# In[11]:


sns.countplot(x='Survived',data=df,hue='Sex')


# In[12]:


df.info()


# In[13]:


df.describe()


# In[15]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


df=df[df['Fare']<400]


# In[25]:


train_df=df


# In[27]:


titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]


# In[28]:


titanic.isnull().any()


# In[29]:


titanic["Age"].fillna(titanic["Age"].mean(),inplace=True)


# In[30]:



# Mapping Sex
titanic['Sex'] = titanic['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[31]:


# The columns that we will be making predictions with
X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

#The column that we will be making predictions on
y = titanic['Survived']


# In[32]:


### Split data randomly into 70% training and 30% test ###

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[33]:


model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)


# In[35]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# In[36]:


predicted = model2.predict(X_test)
print("Accuracy for test data set:\n")
print (format(metrics.accuracy_score(y_test, predicted) * 100,'.2f'), '%.')


# In[37]:


param_test1 = {
 'max_depth': range(2, 10),   
 'min_samples_split': [3, 5, 7, 10],
 'min_samples_leaf': [3, 5, 7, 10]
}

grid_result = GridSearchCV(DecisionTreeClassifier(), param_grid=param_test1, cv=10, n_jobs=-1, verbose=1)
grid_result.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[38]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[39]:


predictions=predictions = dtree.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report,confusion_matrix


# In[41]:


print(classification_report(y_test,predictions))


# In[42]:


print(confusion_matrix(y_test,predictions))


# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


rfc = RandomForestClassifier(n_estimators=600)


# In[45]:


rfc.fit(X_train, y_train)


# In[46]:


rfc_pred = rfc.predict(X_test)


# In[47]:


print(classification_report(y_test,rfc_pred))


# In[48]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:




