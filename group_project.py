# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:34:37 2019

@author: LWQIN
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#load data
df = pd.read_csv('kc_house_data.csv')

"""
###EDA
"""
#Check the types of all the column
df.shape #21613 rows, 21 columns
df.dtypes #all numeric

#check for missing values
df.isna().sum() #no missing values

#Convert zipcode into categorical as it represent location not numeric values
#Convert waterfront to categorical as well (according to data table)
df['zipcode'] = df['zipcode'].astype('category',copy=False)
df['waterfront'] = df['waterfront'].astype('category',copy=False)
df.dtypes

#removing id as it does not contribute to the analysis
df.drop(columns = 'id', inplace = True)
df.shape

#the date column is not formatted correctly e.g.20141013T000000
#to extract only the year from the date
for i in range(0,len(df)): #require quite some time to loop through
    df.iloc[i,0] = df.iloc[i,0][0:4] 

#convert date from string to int
df['date'] = df['date'].astype('int64', copy=False)
df.dtypes
columns = df.columns.values
np.delete(columns,1) #removed price

#scatter plot of all variables against price
pp = sns.pairplot(data=df,
                  y_vars=['price'],
                  x_vars=columns
                  )

#boxplot
plt.figure(figsize=(15,8))
sns.boxplot(df['bedrooms'],df['price'])

plt.figure(figsize=(15,8))
sns.boxplot(df['bathrooms'],df['price'])

plt.figure(figsize=(15,8))
sns.boxplot(df['grade'],df['price'])

#correlation between all numeric variables
corr_matrix = df.corr()

#Filter only variables which has correlation > 0.5 with price (target)
filter_corr_matrix = corr_matrix.loc['price',][corr_matrix.loc['price',] > 0.5]
filter_corr_matrix.pop('price') #removing price
highly_correlated_variables = filter_corr_matrix.index.values

#remove highly correlated independent variables
highly_correlated_variables_corr = df[highly_correlated_variables].corr()
sns.heatmap(highly_correlated_variables_corr) #perhaps can try using all 5 variables

#histogram of price
plt.hist(df['price'],bins =50) #not normally distributed, right skewed
plt.xticks(rotation=90)

#check distribution of each selected variable
plt.hist(df['bathrooms']) #right skewed
plt.hist(df['sqft_living']) #right skewed
plt.hist(df['grade']) #left skewed
plt.hist(df['sqft_above']) #right skewed
plt.hist(df['sqft_living15']) #right skewed

#prepare x and y dataset
#transform x and y to make them normally distirbuted for better accuracy
X = df[highly_correlated_variables]

# transform X['bathrooms'] by log(1+x) for right skewed data
X['bathrooms'] = np.log1p(X['bathrooms']) 

# transform X['sqft_living'] by log(1+x) for right skewed data
X['sqft_living'] = np.log1p(X['sqft_living']) 
 
# transform X['grade'] by multiplying to the power of 1.5 for right skewed data,
#can be also square or cube, but for this case, 1.5 works best
X['grade'] = X['grade']**1.5 

# transform X['sqft_above'] by log(1+x) for right skewed data
X['sqft_above'] = np.log1p(X['sqft_above']) 

# transform X['sqft_living15'] by square rooting for right skewed data, log(1+x) does not work for this variable
X['sqft_living15'] = np.sqrt(X['sqft_living15']) 

 # transform y by log(1+x) for right skewed data
y = df['price']
y = np.log1p(y)

#reference for data transformation http://seismo.berkeley.edu/~kirchner/eps_120/Toolkits/Toolkit_03.pdf

plt.hist(X['bathrooms']) #looks more normally distributed
plt.hist(X['sqft_living']) #looks more normally distributed
plt.hist(X['grade']) #looks more normally distributed
plt.hist(X['sqft_above']) #looks more normally distributed
plt.hist(X['sqft_living15']) #looks more normally distributed
plt.hist(y) #looks more normally distributed

#Adding categorical variables into X
X['zipcode'] = df['zipcode']
X['waterfront'] = df['waterfront']


#One hot encode categorical value 
X['zipcode'].value_counts()
X_with_dummy = pd.get_dummies(X,drop_first=True)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X_with_dummy, y, test_size=0.3, random_state=101)

#linear regression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.intercept_
lm.score(X_train,y_train) #R2 score for training
coeff_df = pd.DataFrame(lm.coef_,X_with_dummy.columns,columns=['Coefficient'])
coeff_df

#Predict the house price and evaluate the model
predictions = lm.predict(X_test)
predictions = np.expm1(predictions) #to inverse the log
y_test = np.expm1(y_test)
plt.scatter(y_test,predictions)

#Evaluate the result
#define a function for MAPE as there is no built-in function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MAPE:', mean_absolute_percentage_error(y_test, predictions)) #14.77%
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R2:', metrics.r2_score(y_test, predictions)) #0.87!