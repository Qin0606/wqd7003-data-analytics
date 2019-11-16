# wqd7003-data-analytics
Group Project

This repo is for WQD7003 Data Analytics Group Project.

Dataset used is taken from https://www.kaggle.com/harlfoxem/housesalesprediction

Code is in Python.

Aim: Predict housing price based on given variables using linear regression

Process:
1. Load data
2. Check the data types of all the column
3. Check for missing values
4. Convert zipcode column into categorical as it represent location not numeric values
5. Convert waterfront column to categorical as well (according to data table)
6. Removing id as it does not contribute to the analysis
7. The date column is not formatted correctly e.g.20141013T000000, extract only the year from the date
8. Convert date from string to int
9. Data visualization: Scatter plot of all variables against price
10. Data visualization: boxplot for bedrooms vs price, bathrooms vs price, grade vs price
11. Compute correlation between all numeric variables
12. Selected 5 out of 17 features which are highly correlated with price. e.g. >0.5
13. None of the 5 features had correlation >0.9 among themselves, thus all 5 features will be used.
14. Check distribution of all 5 features and also price. Notice all of them are not normally distributed.
15. Transform all 5 features and price to make them normally distributed. This is to maximize accuracy and to fulfill linear regression's normality assumption.
16. Combine the 5 selected features together with categorical variables (zipcode and waterfront)
17. One hot encode categorical variables.
18. Train test split (70:30) seed = 101
19. Fit linear regression model to X_train and y_train. Get the R2 score of the model.
20. Predict new value using X_test.
21. Inverse the transformation on the new value and y_test to revert it back to the original value.
22. Plot scatter plot of actual vs predicted.
23. Evaluate the result using Mean Absolute Error, Mean Absolute Percentage Error, Mean Squared Error, Root Mean Squared Error and R2)
