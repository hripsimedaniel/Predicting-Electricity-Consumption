#!/usr/bin/env python
# coding: utf-8

# # Predicting Electricity Consumption - Classification Model
# 
# The dataset used for this project is located here: https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata
#         
# The goal of the project is to build a model that predicts the electricity consumption (the 'KWH' field). 
# 
# In this project, Classification model - RandomForestClassifier - is applied to predict consumption, and GridSearch is used to find optimized hyperparameters.  
# 
# During the preprocessing stage, I split the values of the following columns: 'ACROOMS', 'HEATROOM', 'MONEYPY', 'KWH' (same as 'label') into groups (categories) to make them more suitable for Classification model. All the selected variables/features were sorted into Numerical and Categorical. Categorical columns were OneHotCoded.  
# 
# The performance of the Classification model can be improved if feature selection analysis is conducted more thouroughly, and NaN/missing values are better handled (these values were coded as '-2' in this dataset ). Due to the time limit, I had to spend little time on these two mentioned processes. 
# 
# The result achieved - prediction accuracy of 69.73%.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[2]:


data_files = pd.read_csv("recs2009_public.csv")
pd.set_option("display.max_columns", None)
data_files.head()


# In[3]:


# The following variables were selected to be used in the model.  
#Below feature importance analysis is conducted and plotted.

df_data = data_files[['DOEID', 'TYPEHUQ', 'NWEIGHT', 'HDD30YR', 'CDD30YR', 
'Climate_Region_Pub', 'AIA_Zone', 'YEARMADERANGE', 'WALLTYPE', 
'ROOFTYPE', 'BEDROOMS', 'TOTROOMS', 'HEATHOME', 'FUELHEAT',
'MAINTHT', 'EQUIPAGE', 'REPLCHT', 'EQUIPAUX', 'HEATROOM',
'THERMAIN', 'MOISTURE', 'H2OTYPE1', 'FUELH2O',
'WHEATAGE', 'AIRCOND', 'COOLTYPE', 'AGECENAC', 'ACROOMS',
'NOTMOIST', 'TOTSQFT','MONEYPY', 'HDD50', 'CDD80', 'KWH']]

# df_data.head(20)


# In[4]:


# The following variables are plotted and visualized to analyse the data:
#'DOEID', 'TYPEHUQ', 'NWEIGHT', 'HDD30YR', 'CDD30YR','MONEYPY',  'ACROOMS', 'HEATROOM', 'CDD80'

x_val = range(df_data.shape[0])
# plt.plot(x_val, df_data['DOEID'])
# df_data['CDD80'].value_counts()
plt.plot(x_val, df_data['CDD80'].sort_values())
plt.show()


# In[5]:


#Splitting the values of 'ACROOMS' column into groups (categories) and plotting them
df_data['ACROOMS'].value_counts()
plt.plot(x_val, df_data['ACROOMS'].sort_values())
plt.show()

cut_offs =[-2, 5, 10]
df_data.loc[(df_data['ACROOMS']>=cut_offs[0]) & (df_data['ACROOMS']<cut_offs[1]), 'ACROOMS']=1
df_data.loc[(df_data['ACROOMS']>=cut_offs[1]) & (df_data['ACROOMS']<cut_offs[2]), 'ACROOMS']=2
df_data.loc[df_data['ACROOMS']>=cut_offs[2], 'ACROOMS']=3

df_data['ACROOMS'].value_counts()
plt.plot(x_val, df_data['ACROOMS'].sort_values())
plt.show()


# In[6]:


#Splitting the values of 'HEATROOM' column into groups (categories) and plotting them
df_data['HEATROOM'].value_counts()
plt.plot(x_val, df_data['HEATROOM'].sort_values())
plt.show()

cut_offs =[-2, 5, 10]
df_data.loc[(df_data['HEATROOM']>=cut_offs[0]) & (df_data['HEATROOM']<cut_offs[1]), 'HEATROOM']=1
df_data.loc[(df_data['HEATROOM']>=cut_offs[1]) & (df_data['HEATROOM']<cut_offs[2]), 'HEATROOM']=2
df_data.loc[df_data['HEATROOM']>=cut_offs[2], 'HEATROOM']=3

df_data['HEATROOM'].value_counts()
plt.plot(x_val, df_data['HEATROOM'].sort_values())
plt.show()


# In[7]:


#Splitting the values of 'MONEYPY' column into groups (categories) and plotting them
df_data['MONEYPY'].value_counts()
plt.plot(x_val, df_data['MONEYPY'].sort_values())
plt.show()

cut_offs =[5, 10, 15, 20]
df_data.loc[(df_data['MONEYPY']<cut_offs[0]), 'MONEYPY']=0
df_data.loc[(df_data['MONEYPY']>=cut_offs[0]) & (df_data['MONEYPY']<cut_offs[1]), 'MONEYPY']=1
df_data.loc[(df_data['MONEYPY']>=cut_offs[1]) & (df_data['MONEYPY']<cut_offs[2]), 'MONEYPY']=2
df_data.loc[(df_data['MONEYPY']>=cut_offs[2]) & (df_data['MONEYPY']<cut_offs[3]), 'MONEYPY']=3
df_data.loc[df_data['MONEYPY']>=cut_offs[3], 'MONEYPY']=4

df_data['MONEYPY'].value_counts()
plt.plot(x_val, df_data['MONEYPY'].sort_values())
plt.show()


# In[8]:


#Plotting the values of 'KWH' ('label') column
x_val = range(df_data.shape[0])
plt.plot(x_val,df_data['KWH'].sort_values())
plt.show()


# In[9]:


#Splitting the values of 'KWH' column into groups (categories) and renaming the column into 'label'
cut_offs =[4000, 10000]
df_data['label']=0
df_data.loc[(df_data['KWH']>=cut_offs[0]) & (df_data['KWH']<cut_offs[1]), 'label']=1
df_data.loc[df_data['KWH']>=cut_offs[1], 'label']=2

df_data[:5]


# In[10]:


#Dropping 'DOEID' and 'KWH' columns
df_data.drop(['DOEID', 'KWH'],axis=1,inplace=True)
df_data[:5]


# In[11]:


#Sorting the variables into Numerical and Categorical

num_cols = ['NWEIGHT', 'HDD30YR', 'CDD30YR', 'TOTSQFT', 'HDD50', 'CDD80','label']
df_data[num_cols][:5]
# Need to OneHotEncode the following columns: 
categ_cols = ['TYPEHUQ', 'Climate_Region_Pub', 'AIA_Zone', 'YEARMADERANGE', 'WALLTYPE', 'ROOFTYPE', 
'HEATHOME', 'FUELHEAT', 'MAINTHT', 'EQUIPAGE', 'REPLCHT', 'EQUIPAUX', 'THERMAIN', 
'MOISTURE', 'H2OTYPE1', 'FUELH2O',
'WHEATAGE', 'AIRCOND', 'COOLTYPE', 'AGECENAC', 'HEATROOM', 'ACROOMS','NOTMOIST', 'MONEYPY']
df_data[categ_cols][:5]


# In[12]:


#OneHotEncoding of Categorical Columns
df_onecode = df_data[categ_cols].copy()
df_onecode.shape
for cur_col in df_onecode.columns:
    dummy_col = pd.get_dummies(df_onecode[cur_col], prefix=cur_col+'_')
#     dummy_col[:5]
    df_onecode = pd.merge(left=df_onecode, right=dummy_col, left_index=True, right_index=True)
#     df_onecode[:5]

df_onecode.shape
# df_onecode[:5]
df_onecode.drop(categ_cols,axis=1,inplace=True)
df_onecode.shape
df_onecode[:5]


# In[13]:


#Merging new one-hot encoded columns with the numerical dataset
df_data.drop(categ_cols,axis=1,inplace=True)
df_data[:5]
df_data = pd.concat([df_onecode, df_data], axis=1)
df_data[:5]


# In[14]:


#Plotting 'label' column
plt.plot(range(df_data.shape[0]),df_data['label'].sort_values())
plt.show()


# In[15]:


#Splitting into train and test sets
y = df_data['label']
X = df_data.drop('label',axis=1)
X.shape
X[:5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)


# In[16]:


#Standardization 
sd = StandardScaler()

X_train = sd.fit_transform(X_train) 
X_test = sd.fit_transform(X_test) 


# In[17]:


#RandomForest Classifier model
start = time.time()
clf = RandomForestClassifier(n_estimators=500, max_depth=12, 
                             class_weight = 'balanced', n_jobs=-1)
clf.fit(X_train, y_train)

print("  ", int(time.time()-start), "Seconds to execute")


# In[18]:


predict = clf.predict(X_test)
proba = clf.predict_proba(X_test)
cm = confusion_matrix(y_test, predict)

print('ACC: ', accuracy_score(y_test, predict).round(2))
print('Confusion Matrix: \n', cm, '\n')
print(classification_report(y_test, predict))


# In[19]:


#Checking features importance 
train_columns = X.columns
feature_importances = pd.DataFrame(clf.feature_importances_,index = np.array(train_columns),
                                    columns=['importance']).sort_values('importance', ascending=False)
# printing first 10 features
feature_importances[:10]


# In[20]:


#Plotting Features Importance Values (first 10 features)
def plot_feature_importance(feature_importances):
    labels = feature_importances[:10].index.to_list()
    values = feature_importances['importance'][:10].values
    fig = plt.subplots(figsize=(10, 8))
    plt.bar(labels,
            values,
            align='center',
            color='red')
    plt.xticks(rotation = 45)
    plt.title('Feature Importance')
    plt.ylabel('importances')
    
plot_feature_importance(feature_importances)


# In[21]:


# finding hyperparameters with GridSearchCV 
param_grid = {'n_estimators':range(200,500,700), 
          'max_depth':range(12,15,21),
         } 
params = {'n_estimators':500, 'max_depth':12,
          'criterion' :'entropy', 'warm_start': True,
          'class_weight': 'balanced'
         } 

gs = GridSearchCV(estimator = RandomForestClassifier(params), 
                  param_grid = param_grid, 
                  cv=5,
                  verbose=10,
                  n_jobs=-1)


# In[22]:


start = time.time()
gs.fit(X_train,y_train)
print("  ", int(time.time()-start), "Seconds to execute")


# In[23]:


gs.best_estimator_
gs.best_params_
gs.best_score_


# In[ ]:





# In[ ]:




