#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:11:57 2021

@author: Sundeep Maddu
"""
import pandas            as pd
import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt
%matplotlib inline 
import statsmodels.api   as sm
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing   import LabelEncoder
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split # Sklearn package's randomized data splitting function
from sklearn.metrics         import mean_squared_error
from sklearn.preprocessing   import StandardScaler

from imblearn.over_sampling  import SMOTE

# Removes the limit from the number of displayed columns and rows.
# This is so I can see the entire dataframe when I print it
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_rows', 200)

# Read the data from CSV - Start
df = pd.read_csv('/Users/sundeep/Learnings/1_PGP_DSBA/Project/6_Cars4U/used_cars_data.csv',index_col=0)
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in the dataset.')  # f-string
# I'm now going to look at 10 random rows
# I'm setting the random seed via np.random.seed so that
# I get the same random results every time
np.random.seed(1)
df.sample(n=5)
# Read the data from CSV- Complete

df.info()
df.isnull().sum().sort_values(ascending=False)
df.isnull().any()
df.describe()

df['Mileage'] = df['Mileage'].str.strip(' kmpl')
df['Mileage'] = df['Mileage'].str.strip(' km/kg')
df['Engine']  = df['Engine'].str.strip(' CC')
df['Power']   = df['Power'].str.strip(' bhp')
df['New_Price']   = df['New_Price'].str.strip(' Lakh')

# df['Mileage']=df['Mileage'].str.replace('kmpl','')
# df['Mileage']=df['Mileage'].str.replace('km/kg','')
# df['Engine']=df['Engine'].str.replace('CC','')
# df['Power']=df['Power'].str.replace('bhp','')
# df['Mileage']=df['Mileage'].str.replace('null','0')
# df['Engine']=df['Engine'].str.replace('null','0')
# df['Power']=df['Power'].str.replace('null','0')

# Name = df['Name'].str.split(" ", n = 5, expand = True)
# Name.head(150)
# df.drop(['Name'], axis=1, inplace=True)
# df["Name_1"]= Name[0]   
# df["Name_2"]= Name[1] 
# del Name  # don't need to do this but can keep things tidy


df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Engine']  = pd.to_numeric(df['Engine'], errors='coerce')
df['Power']   = pd.to_numeric(df['Power'], errors='coerce')
df['Seats']   = pd.to_numeric(df['Seats'], errors='coerce')
df['New_Price']   = pd.to_numeric(df['New_Price'], errors='coerce')

# Name = df['Name'].str.split(" ", n = 5, expand = True)
# Name.head(150)
# df.drop(['Name'], axis=1, inplace=True)
# df["Name_1"]= Name[0]   
# df["Name_2"]= Name[1] 
# del Name  # don't need to do this but can keep things tidy



# dummy = pd.get_dummies(df[['Location','Fuel_Type','Transmission','Owner_Type']], drop_first=True)
# dummy
# df = pd.concat([df, dummy], axis=1)
# df = df.drop(['Location','Fuel_Type','Transmission','Owner_Type','New_Price','Name'],axis=1)

df = pd.get_dummies(df, columns=['Transmission'], drop_first=True)
df['Fuel_Type'] = df['Fuel_Type'].map({'Diesel':1,'Petrol':2,'CNG':3,'LPG':4})
df['Owner_Type'] = df['Owner_Type'].map({'First':1,'Second':2,'Third':3,'Fourth & Above':4})
df['Location'] = df['Location'].map({'Mumbai':1,'Pune':2,'Chennai':3,'Coimbatore':4,'Hyderabad':5,'Jaipur':6,'Kochi':7, 'Kolkata':8, 'Delhi':9, 'Bangalore':10, 'Ahmedabad':11})

df.drop(['Name'],axis = 1, inplace = True)
df.drop(['New_Price'],axis = 1, inplace = True)

# we will replace missing values in every column with its medain
medianFiller = lambda x: x.fillna(x.median())
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
# numeric_columns
df[numeric_columns] = df[numeric_columns].apply(medianFiller,axis=0)
df

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df['Engine'] = imputer.fit_transform(df['Engine'].values.reshape(-1, 1))
df['Power'] = imputer.fit_transform(df['Power'].values.reshape(-1, 1))
df['Mileage'] = imputer.fit_transform(df['Mileage'].values.reshape(-1, 1))
df['Seats'] = imputer.fit_transform(df['Seats'].values.reshape(-1, 1))
df['Price'] = imputer.fit_transform(df['Price'].values.reshape(-1, 1))
df['Fuel_Type'] = imputer.fit_transform(df['Fuel_Type'].values.reshape(-1, 1))

df = np.log1p(df)
df = df.replace('?', np.nan)

# df['Engine']=df['Engine'].astype(float)
# df['Power']=df['Power'].astype(float)
# df['Mileage']=df['Mileage'].astype(float)

# df.loc[df.Engine==0,'Engine']=np.NaN
# df.loc[df.Power==0,'Power']=np.NaN


# df['Engine']=df['Engine'].fillna(df['Engine'].mean())
# df['Power']=df['Power'].fillna(df['Power'].mean())
# df['Seats']=df['Seats'].fillna(df['Seats'].mean())


# df.Fuel_Type.unique()
# df.Location.unique()

# le = LabelEncoder()
# en_df = df.apply(le.fit_transform)
# en_df.head()

Y = df['Price'].values
X = df.drop(columns = {'Price'}).values

#splitting the data in 70:30 ratio of train to test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30 , random_state=1)

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# # y_train = sc_y.fit_transform(y_train)
# Y = np.squeeze(sc_y.fit_transform(Y.reshape(-1, 1)))


#intialise the model to be fit and fir the model on the train data
regression_model = LinearRegression(fit_intercept=True)
regression_model.fit(X_train, y_train)

print('The coefficient of determination R^2 of the prediction on Train set', regression_model.score(X_train, y_train))
print('The coefficient of determination R^2 of the prediction on Test set',regression_model.score(X_test, y_test))

# import statsmodels.api as sm

# X = sm.add_constant(X)
# linearmodel = sm.OLS(Y, X).fit()
# predictions = linearmodel.predict(X) 
# print_model = linearmodel.summary()
# print(print_model)








