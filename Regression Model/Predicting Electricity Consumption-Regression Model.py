#!/usr/bin/env python
# coding: utf-8

# # Predicting Electricity Consumption - Regression Model
# 
# The dataset used for this project is located here: https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata 
#         
# The goal of the project is to build a model that predicts the electricity consumption (the 'KWH' field). 
# 
# The following Regression models are applied in this project: RandomForestRegressor, DecisionTreeRegressor, ExtraTreeRegressor, and LinearRegression.  
# 
# During the preprocessing stage, the dimentionality reduction of features is conducted with Truncated SVD.
# 
# The performance of the Regression model can be improved if feature selection analysis is conducted more thouroughly, and NaN/missing values (these values were coded as '-2' in this dataset) are better handled. Due to the time limit, I had to spend little time on these two mentioned processes. 

# In[1]:


import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD


# In[2]:


#Examine the raw dataset
data_files = pd.read_csv("recs2009_public.csv")
pd.set_option("display.max_columns", None)
data_files.head()


# In[3]:


#The following variables were selected to be used in the model. 
df_data = data_files[['DOEID', 'CDD30YR', 'BEDROOMS', 'TOTROOMS', 'FUELH2O','TOTSQFT', 'HDD50', 'KWH']]

# df_data.head(20)


# In[4]:


#Visualizing label - the 'KWH' field.
x_val = range(df_data.shape[0])
plt.plot(x_val,df_data['KWH'].sort_values())
plt.show()


# In[5]:


#Replacing 'KWH' field with 'label' field
df_data['label']=df_data['KWH'].copy()
df_data[:5]


# In[6]:


#Dropping KWH and DOEID fields
df_data.drop(['DOEID', 'KWH'],axis=1,inplace=True)
df_data[:5]


# In[7]:


#Splitting into train and test sets
y = df_data['label']
X = df_data.drop('label',axis=1)
X.shape
X[:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)


# In[8]:


#Function for Metrics
def train(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)), '\n') 
    return y_pred

# Dimentionality reduction with Truncated SVD
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

#Run 4 models to compare metrics
clf = LinearRegression()
print('Linear reg')
y_pred_lr = train(clf)
clf = RandomForestRegressor(max_depth= 5, max_features= 'sqrt', min_samples_split= 2, n_estimators = 5, random_state =42)
print('Random Forest')
y_pred_rf = train(clf)
clf = DecisionTreeRegressor(min_samples_split = 40, max_depth = 15, random_state =42)
print('Decision Tree')
y_pred_dt = train(clf)
clf = ExtraTreeRegressor(min_samples_leaf = 40, min_samples_split = 15, random_state=42)
print('Extra Tree')
y_pred_et = train(clf)


df_result = pd.DataFrame(y_test)
df_result['pred_lr'] = y_pred_lr.astype(int)
df_result['pred_rf'] = y_pred_rf.astype(int)
df_result['pred_dt'] = y_pred_dt.astype(int)
df_result['pred_et'] = y_pred_et.astype(int)


# In[9]:


df_result.sample(10)


# In[ ]:




