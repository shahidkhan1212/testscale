#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model  import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import seaborn as sns

import pickle

import warnings
warnings.filterwarnings('ignore')


# In[83]:


data = pd.read_csv('heart_disease.csv')
data.head()


# In[84]:


data.describe()


# In[85]:


data.isnull().sum()


# In[86]:


data.info()


# In[87]:


# replacing zero values with the mean of the column
data['Unnamed: 0']=data['Unnamed: 0'].replace(0,data['Unnamed: 0'].mean())




# In[88]:


# lets see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='yellow')
plotnumber = 1

for column in data:
    if plotnumber<=15:
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.tight_layout()


# In[89]:


df_features = data.drop('target',axis=1)


# In[90]:


# Visualize the outliers using boxplot
plt.figure(figsize=(30,35))
graph = 1

for column in df_features:
    if graph<=15:
        plt.subplot(5,4,graph)
        ax=sns.boxplot(data= df_features[column])
        plt.xlabel(column,fontsize=15)
        
    graph+=1
plt.tight_layout()


# In[91]:


data.shape


# In[92]:


# find the IQR (inter quantile range) toidentify outliers

# 1st Quantile
q1 = data.quantile(0.25)

# 3rd Quantile
q3 = data.quantile(0.75)

#IQR
iqr = q3 - q1


# In[93]:


q1


# In[94]:


# validating one outlier

tres_high = (q3.trestbps + (1.5 * iqr.trestbps))
tres_high


# In[95]:


# check the index which have higher values
np_index = np.where(data['trestbps'] > tres_high)
np_index


# In[96]:


# Drop the index which we found in the above cell

data = data.drop(data.index[np_index])
data.shape


# In[97]:


data.reset_index()


# In[98]:


# validating one outlier

chol_high = (q3.chol + (1.5 * iqr.chol))
chol_high


# In[99]:


# check the index which have higher values
index = np.where(data['chol'] > chol_high)

data =data.drop(data.index[index])

data.reset_index()


# In[70]:


# validating one outlier

fbs_high = (q3.fbs + (1.5 * iqr.fbs))
fbs_high


# check the index which have higher values
index1 = np.where(data['fbs'] > fbs_high)

data =data.drop(data.index[index1])

data.reset_index()


# In[100]:


old_high = (q3.oldpeak + (1.5 * iqr.oldpeak))
old_high

# check the index which have higher values
index2 = np.where(data['oldpeak'] > old_high)

data =data.drop(data.index[index2])
print(data.shape)

data.reset_index()


# In[101]:


ca_high = (q3.ca + (1.5 * iqr.ca))
ca_high

# check the index which have higher values
index3 = np.where(data['ca'] > ca_high)

data =data.drop(data.index[index3])
print(data.shape)

data.reset_index()


# In[102]:


# validating one outlier

thalach = (q1.thalach - (1.5 * iqr.thalach))
thalach

# check the index which have higher values
index4 = np.where(data['thalach'] < thalach)

data =data.drop(data.index[index4])
print(data.shape)

data.reset_index()


# In[103]:


# validating one outlier

thal = (q1.thal - (1.5 * iqr.thal))
thal

# check the index which have higher values
index5 = np.where(data['thal'] < thal)

data =data.drop(data.index[index5])
print(data.shape)

data.reset_index()


# In[104]:


# lets see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='yellow')
plotnumber = 1

for column in data:
    if plotnumber<=15:
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.tight_layout()


# In[105]:


X= data.drop(columns = ['target'])
y = data['target']


# In[106]:


# 
plt.figure(figsize=(15,20))
plotnumber = 1

for column in X:
    if plotnumber<=15:
        ax = plt.subplot(3,5,plotnumber)
        sns.stripplot(x=y,y=X[column],hue=y)
        
        
    plotnumber+=1
plt.show()


# In[107]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[108]:


X_scaled.shape[1]


# In[109]:


vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif['features'] = X.columns

vif


# In[110]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)
y_train.head()


# In[111]:


log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)


# In[112]:


y_pred = log_reg.predict(x_test)


# In[113]:


y_pred


# In[114]:


log_reg.predict_proba(x_test)


# In[115]:


# confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[116]:


#Model Accuracy
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[ ]:





# In[117]:


from sklearn.metrics import classification_report


# In[118]:


print (classification_report(y_test, y_pred))


# In[119]:


#ROC CURVE
fpr,tpr, thresholds = roc_curve(y_test,y_pred)


# In[120]:


# thresholds[0] means no instances predicted (it should be read from 0 - max)
print ('Threshold =', thresholds)
print('True Positive rate = ', tpr)
print ('False Positive rate = ',fpr)


# In[121]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1],[0, 1], color='darkblue', linestyle='--')

plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[122]:


# How much area it is covering(AUC)
auc_score = roc_auc_score(y_test,y_pred)
print(auc_score)


# In[ ]:




