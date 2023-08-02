#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np


# In[13]:


df = pd.read_csv('loan_data_set.csv')


# In[14]:


df


# In[15]:


df.isna().sum()


# In[16]:


df_new = df.dropna()


# In[17]:


df_new.isna().sum()


# In[23]:


df_new['Dependents'].unique()


# In[24]:


df_new['Gender_dummy'] = 0
df_new['Married_dummy'] = 0
df_new['Dependents_dummy'] = 0
df_new['Education_dummy'] = 0
df_new['Self_Employed_dummy'] = 0
df_new['Property_Area_dummy'] = 0



df_new['Gender_dummy'] = df_new['Gender'].apply(lambda x: 1.0 if x == 'Male' else 0.0)
df_new['Married_dummy'] = df_new['Married'].apply(lambda x: 1.0 if x == 'Yes' else 0.0)
df_new['Education_dummy'] = df_new['Education'].apply(lambda x: 1.0 if x == 'Graduate' else 0.0)
df_new['Self_Employed_dummy'] = df_new['Self_Employed'].apply(lambda x: 1.0 if x == 'Yes' else 0.0)


property_area_mapping = {
    'Urban':1.0,
    'Rural':0.0,
    'Semiurban':0.5    
}
df_new['Property_Area_dummy'] = df_new['Property_Area'].apply(lambda x: property_area_mapping.get(x, 0.0))

Dependents_mapping = {
    '0':0.0,
    '1':1.0,     
    '2':2.0,
    '3+':3.0
    
}
df_new['Dependents_dummy'] = df_new['Dependents'].apply(lambda x: Dependents_mapping.get(x,0.0))


# In[25]:


df_new.head()


# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

feature_cols = ['Gender_dummy','Married_dummy','Dependents_dummy','Education_dummy','Self_Employed_dummy','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area_dummy']

x = df_new[feature_cols]
y = df_new.Loan_Status # Target variablefrom sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)


# In[40]:



from sklearn.linear_model import LogisticRegression
import pickle

logreg = LogisticRegression(random_state = 16)

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

pickle.dump(logreg, open('logistic_model.pkl', 'wb'))


# In[44]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the model on the test data: {accuracy:.2f}")

