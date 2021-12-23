"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Mon Mar 29 18:19:14 2021

@author: Sundeep Maddu
"""

import os
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
# import missingno         as msno
# import pandas_profiling  as pdp

from sklearn.preprocessing   import LabelEncoder
from matplotlib.offsetbox    import AnchoredText
from sklearn.ensemble        import ExtraTreesClassifier
from sklearn.impute          import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import accuracy_score, classification_report, roc_curve,precision_recall_curve, auc,confusion_matrix
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm             import SVC
from sklearn.impute          import KNNImputer
# from xgboost                 import XGBClassifier
# from catboost                import CatBoostClassifier
from imblearn.over_sampling  import SMOTE


# Current working drectory using OS library #
path = os.getcwd() 
path
# Current working drectory using OS library #

# Change working drectory using OS library #
os.chdir('/Users/sundeep/Learnings/1_PGP_DSBA/Project/5_HeartStroke/')
path = os.getcwd() 
path
# Change working drectory using OS library #


df = pd.read_csv('healthcare-dataset-stroke-data.csv') # read input data into data frame
df.head()

# axis=0 means along "indexes". It's a row-wise operation.
# axis=1 means along "columns". It's a column-wise operation.
df = df.drop('id',axis = 1) 
# df.dropna(inplace = True)

df.info()
df.shape
df = df.drop_duplicates() # To remove duplicates
df.shape

df.isnull().sum() # To check the null values across each column

##### PIE Plot #####
labels = df['stroke'].value_counts(sort = True).index
sizes  = df['stroke'].value_counts(sort = True)
colors = ['lightblue','red']
explode = (0.05,0)
plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,
        autopct='%1.1f',shadow=True,startangle=90)
plt.title('No of Strokes in the datset')
plt.show()
##### PIE Plot #####

##### Label Encoder - Change the string data to numeric data #####
le = LabelEncoder()
en_df = df.apply(le.fit_transform)
en_df.head()
##### Label Encoder - Change the string data to numeric data #####

##### Function for Plot #####
def plot_hist(col,bins=30,title='',xlabel='',ax=None):
    sns.distplot(col,bins=bins,ax=ax)
    ax.set_title(f'Histogram of{title}',fontsize = 20)
    ax.set_xlabel(xlabel)


fig, axes = plt.subplots(1,3,figsize=(11,7),constrained_layout=True)
plot_hist(df.bmi,
          title='Bmi',
          xlabel="Level of the BMI",
          ax=axes[0])
plot_hist(df.age,
          bins=30,
          title='Age',
          xlabel='Age',
          ax=axes[1])
plot_hist(df.avg_glucose_level,
          title='Serum Creatinine', 
          xlabel='Level of serum creatinine in the blood (mg/dL)',
          ax=axes[2])

plt.show()

##### catplot #####
sns.catplot(y='work_type',hue='stroke',kind='count',
            palette='pastel',edgecolor='0.6',data=df)

plt.figure(figsize=(17,7))
sns.catplot(x="gender", y="stroke", hue="heart_disease", palette="pastel", kind="bar", data=df)
sns.catplot(x="gender", y="stroke", hue="Residence_type", palette="pastel", kind="bar", data=df)
sns.catplot(x="gender", y="stroke", hue="hypertension", palette="pastel", kind="bar", data=df)
plt.show()
##### catplot #####

##### Unique values in the character columns #####
for column in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] :
    print(df[column].unique())
##### Unique values in the character columns #####

#### Plot using for loop #####
barplot_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
countplot_cols = ['heart_disease', 'hypertension']
boxplot_cols = ['age','avg_glucose_level', 'bmi']

for i, column in enumerate(barplot_cols):
    # print (i ,column)
    sns.barplot(x='stroke',y=column,data=df)
    plt.show()
    
for column in barplot_cols:
    # print (i ,column)
    sns.barplot(x='stroke',y=column,data=df)
    plt.show()
#### Plot using for loop #####    

##### Replace values #####
replace_values = {'Unknown' : 'Never Smoked','formerly smoked': 'smokes'}
df = df.replace({'smoking_status' : replace_values})
##### Replace values #####


# def feature_creation(df):
#     df['age1'] = np.log(df['age'])
#     df['age2'] = np.sqrt(df['age'])
#     df['age3'] = df['age']**3
#     df['bmi1'] = np.log(df['bmi'])
#     df['bmi2'] = np.sqrt(df['bmi'])
#     df['bmi3'] = df['bmi']**3
#     df['avg_glucose_level1'] = np.log(df['avg_glucose_level'])
#     df['avg_glucose_level2'] = np.sqrt(df['avg_glucose_level'])
#     df['avg_glucose_level3'] = np.log(df['avg_glucose_level'])*3
#     for i in ['gender', 'age1', 'age2', 'age3', 'hypertension', 'heart_disease', 'ever_married', 'work_type']:
#         for j in ['Residence_type', 'avg_glucose_level1','avg_glucose_level2', 'avg_glucose_level3', 'bmi1', 'bmi2', 'bmi3','smoking_status']:
#             df[i+'_'+j] = df[i].astype('str')+'_'+df[j].astype('str')
#     return df

# df = feature_creation(df)

#### Correlation of Y with each and every column #####
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type','smoking_status']
correlation_table =[]
for columns in features:
    y = en_df['stroke']
    x = en_df[columns]
    corr = np.corrcoef(x,y)[1][0]
    dict = {'Features':columns,'Correlation Coefficient' : corr,'Feat_type':'numerical'}
    correlation_table.append(dict)
corr_df = pd.DataFrame(correlation_table)
fig = plt.figure(figsize=(10,6),facecolor='#EAECEE')
ax = sns.barplot(x='Correlation Coefficient',y='Features',
                 data=corr_df.sort_values('Correlation Coefficient',ascending=False),
                 palette='viridis',alpha=0.75)
ax.grid()
#### Correlation of Y with each and every column #####

##### Correlation with all variables #####

##### Correlation with all variables #####

##### Extra Tree Classifier #####
x = en_df[features]
y = en_df['stroke']
forest = ExtraTreesClassifier(n_estimators = 250 , random_state = 0)
forest.fit(x,y)  
importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis = 0)
indices = np.argsort(importances)[::-1]

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()

plt.title("Feature importances")
sns.barplot(x=np.array(features)[indices], y=importances[indices], palette="deep",yerr=std[indices])
plt.xticks(range(x.shape[1]), np.array(features)[indices],rotation=60)
plt.xlim([-1, x.shape[1]])
plt.show()
##### Extra Tree Classifier #####


##### SMOTE - Synthetic Minority Oversampling Technique #####
'''
if the raito of data between x and y is not same , accuracy of the model will not be effective
due to that we use SMOTE technique to get the data into equal raito between x and y
'''
# smote = SMOTE()
# x = en_df.iloc[:,:-1]
# y = en_df.stroke

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=10)

# en_df_imputed = en_df
# imputer = KNNImputer(n_neighbors=4, weights="uniform")
# imputer.fit_transform(en_df_imputed)
# en_df_imputed.isnull().sum()

# X , y = en_df_imputed[features],en_df_imputed["stroke"]
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=23)
# sm = SMOTE()
# X_res, y_res = sm.fit_resample(x_train,y_train)

# print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
# print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))

# print('After OverSampling, the shape of train_X: {}'.format(X_res.shape))
# print('After OverSampling, the shape of train_y: {} \n'.format(y_res.shape))

# print("After OverSampling, counts of label '1': {}".format(sum(y_res==1)))
# print("After OverSampling, counts of label '0': {}".format(sum(y_res==0)))
##### SMOTE - Synthetic Minority Oversampling Technique #####





