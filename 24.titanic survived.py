#!/usr/bin/env python
# coding: utf-8

# In[90]:


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


# In[91]:


df = pd.read_csv('titanic_train.csv')
df.head()


# In[92]:


#preprocessing steps
df.shape
df.shape[0]


# In[93]:


df.info()


# In[94]:


df.isnull().sum()


# In[95]:


# percentage null value
(df.isnull().sum()/df.shape[0])*100


# In[96]:


#total null value
df.notnull().sum()


# In[97]:


df.isnull().sum().sum() # total null value


# In[98]:


#total notnull value
df.notnull().sum().sum()


# In[99]:


(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100 # overall percent of null values in our dataset


# In[100]:


sns.heatmap(df.isnull())
plt.show


# In[110]:


df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)


# In[114]:


df['Age'].mean()
df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[115]:


df.info()


# In[116]:


#50% data is null then remove that column and remove useless column
df.drop(columns=['Unnamed: 0','Name','Ticket','Cabin'],inplace=True)


# In[117]:


df.info()


# In[130]:


df.isnull().sum()


# In[131]:


obj = df[['Sex']]


# In[140]:


obj.info()


# In[ ]:





# In[135]:


from sklearn.preprocessing import LabelEncoder


# In[141]:


le = LabelEncoder()
df['Sex']= le.fit_transform(df['Sex'])


# In[142]:


df.head()


# In[132]:


pd.get_dummies(obj).info()


# In[133]:


ohe = OneHotEncoder(drop='first')
ar = ohe.fit_transform(obj).toarray()
ar


# In[134]:


pd.DataFrame(ar,columns=['Sex_male'])


# In[143]:


df.head()


# In[54]:


#collect different type data type
ob = df.select_dtypes(include='object')


# In[50]:


from sklearn.preprocessing import OneHotEncoder


# In[55]:


ohe = OneHotEncoder()
ohe.fit_transform(ob).toarray()


# In[56]:


pd.get_dummies(ob)


# In[58]:


ob.isnull().sum()


# In[60]:


ohe = OneHotEncoder()
ohe.fit_transform(ob).toarray()


# In[62]:


ohe = OneHotEncoder()
ar = ohe.fit_transform(ob).toarray()
ar


# In[63]:


pd.DataFrame(ar,columns=['Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'])


# In[64]:


new = pd.DataFrame(ar,columns=['Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'])


# In[67]:


new.info()


# In[66]:


#50% data is null then remove that column
df.drop(columns=['Sex','Embarked'],inplace=True)
df


# In[68]:


# oin data frame
data = df.join(new)


# In[72]:


data.head(400)


# In[71]:


data.isnull().sum()


# In[144]:


df.isnull().sum()


# In[145]:


df.info()


# In[146]:


df.describe()


# In[147]:


#50% data is null then remove that column
df.drop(columns=['name'],inplace=True)
df


# In[148]:


# lets see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='yellow')
plotnumber = 1

for column in df:
    if plotnumber<=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.tight_layout()


# In[150]:


df_features = data.drop('Survived',axis=1)


# In[159]:


# check outliers
# Visualize the outliers using boxplot
df.boxplot(column='Fare')


# In[153]:


X= df.drop(columns = ['Survived'])
y = df['Survived']


# In[154]:


# 
plt.figure(figsize=(15,20))
plotnumber = 1

for column in X:
    if plotnumber<=9:
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(x=y,y=X[column],hue=y)
        
        
    plotnumber+=1
plt.show()


# In[ ]:





# In[155]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[156]:


X_scaled.shape[1]


# In[157]:


vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif['features'] = X.columns

vif


# In[158]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)
y_train.head()


# In[160]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()


log_reg.fit(x_train,y_train)


# In[174]:


# write one function and call as many as time to check accuracy_score of different models.
def metric_score(clf,x_train,x_test,y_train,y_test,train=True):
    if train:
        y_pred = clf.predict(x_train)
        
        print("\n===============Train Result====================")
        
        print(f"Accuracy Score: {accuracy_score(y_train, y_pred) * 100:2f}%")
        
        
    elif train==False:
        
        pred = clf.predict(x_test)
        
        print("\n===============Test Result====================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:2f}%")
        
        print('\n \n Test Classification Report \n', classification_report(y_test, pred,digits=2))
        


# In[175]:


from sklearn.ensemble import RandomForestClassifier
# initiate randomforestclassifier and train

random_clf = RandomForestClassifier()

# train the model
random_clf.fit(x_train,y_train)


# In[176]:


from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
# call the function and pass dataset to check train and test score
metric_score(random_clf,x_train, x_test,y_train,y_test,train=True) #this is for training score

metric_score(random_clf,x_train, x_test,y_train,y_test,train=False) #this is for testing score



# In[177]:


from sklearn.model_selection import GridSearchCV


# In[178]:


# Random forest classifier
params = {'n_estimators':[13,15],
         'criterion':['entropy','gini'],
           'max_depth':[10,15],
            'min_samples_split':[10,11],        
            'min_samples_leaf':[5,6],
}



# In[179]:


grd = GridSearchCV(random_clf,param_grid=params)
grd.fit(x_train,y_train)

print('best_params = > ', grd.best_params_)


# In[180]:


random_clf = grd.best_estimator_

random_clf.fit(x_train, y_train)


# In[181]:


#call the function and pass dataset to check train test score
# call the function and pass dataset to check train and test score
metric_score(random_clf,x_train, x_test,y_train,y_test,train=True) #this is for training score

metric_score(random_clf,x_train, x_test,y_train,y_test,train=False) #this is for testing score



# In[182]:


# plot ROC/AUC for multiple models without hyperparams tuning.

from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[183]:


#load your data and split it into training and testing sets
X= df.drop(columns = ['Survived'], axis=1)
y = df['Survived']


x_train,x_test,y_train,y_test = train_test_split(X, y,test_size = 0.25,random_state= 52)

#train your models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
kn = KNeighborsClassifier()


# In[184]:


models = {'Logistic Regression': lr, 'Random Forest': rf,'Knn':kn,'Decision Tree':dt}

#calculate the Roc curves and Auc scores for each model
plt.figure(figsize=(8, 6))
for name, model in models.items():                        #Read key and values from the item
    model.fit(x_train, y_train)                           # Each model training
    y_prob = model.predict_proba(x_test)[:,1]             # predict prob of each model
    fpr, tpr, _ = roc_curve(y_test, y_prob)               # Finding false and true positive rate('_' is threshold)
    print('Threshold of ' ,name, _)
    roc_auc = auc(fpr, tpr)                               # auc score of each model captured
    
    
    # plot the ROC curve
    plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(name, roc_auc))
    
#Add labels and legend to the plot
plt.plot([0,1],[0,1], linestyle='--', color='grey', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

#show the plot
plt.show


# In[185]:


import pickle


# In[ ]:




