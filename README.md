# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 
3.LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modulesfrom sklearn.
7.Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HEMAPRASAD N
RegisterNumber:  212222040054
*/

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
1.Placement Data
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/e8beaaac-b4e6-4f85-bdfe-a57232470d57)
2.Salary Data
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/e9bb763b-94d5-474e-aebc-fbdc4fd1fbd7)
3.Checking the null() function
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/87989fd8-7ae7-4a4d-8ab9-e9408f828870)
4.Data Duplicate
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/cb17aaa8-84ff-452f-b44c-a0bc9c1c9cfd)
5.Print Data
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/d5896e59-529e-4203-864a-bda838293806)
6.Data-status
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/7a49641b-079b-499b-9013-7ed43fdb5a00)
7.y_prediction array
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/3fe3eca6-ef62-483d-a9ff-16b5c9810f06)
8.Accuracy Value
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/19833c44-8a0c-4077-8d95-2008279ea6d4)
9.Confusion Array
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/e580b833-f34b-47b3-9235-e485d6b1a9a4)
10.Classification Report
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/74aa7ce1-c797-45ac-8884-c95bb60855f9)
11.Prediction of LR
![image](https://github.com/Hemaprasad-N/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135933397/813c226a-e019-4503-8de1-d72348483d5b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
