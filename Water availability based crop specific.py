#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pickle


# In[50]:


df = pd.read_csv(r'C:\Users\Admin\Desktop\Crop\cropdata.csv')


# In[52]:


df.tail()


# In[53]:


df.size


# In[54]:


df.shape


# In[55]:


df.columns


# In[56]:


df['crop'].unique()


# In[57]:


df.dtypes


# In[58]:


df['crop'].value_counts()


# In[59]:


sns.heatmap(df.corr(),annot=True)


# In[60]:


#Seperating features and target label
features = df[['Temperature','PH','Rainfall','Phosphorous','Nitrogen','Potash']]
target = df['crop']


# In[61]:


##accuracy and model name
acc = []
model = []


# In[62]:


# Splitting into train and test data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[63]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
acc_score = []
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
for i in n_estimators:
    RF = RandomForestClassifier(n_estimators=i, random_state=2)
    RF.fit(Xtrain,Ytrain)
    predicted_values = RF.predict(Xtest)

    x = metrics.accuracy_score(Ytest, predicted_values)
    acc_score.append(x) 
    
plt.plot(n_estimators, acc_score, 'b', label='Accuracy')
plt.ylabel('Accuracy score')
plt.xlabel('n_estimators')
plt.show()                      


# In[64]:


acc_score


# In[65]:


RF = RandomForestClassifier(n_estimators=16, random_state=2)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)                         #accuracy
model.append('RF')                   #model name
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[66]:


plt.figure(figsize=[10,1],dpi = 100)

plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[20]:


score = cross_val_score(RF,features,target,cv=5)
score


# In[21]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# In[23]:


#['State_Name', 'District_Name', 'Season', 'Temperature', 'PH',   'Rainfall', 'Phosphorous', 'Nitrogen', 'Potash', 'crop']
data = np.array([[25,6,30,53,23,12,]])
prediction = RF.predict(data)
print(prediction)


# In[ ]:




