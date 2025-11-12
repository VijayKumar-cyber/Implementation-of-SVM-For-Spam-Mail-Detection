# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Collect and preprocess** the email dataset (clean text, remove stopwords, and convert to numerical features using TF-IDF or Bag of Words).
2. **Split** the dataset into training and testing sets.
3. **Train** the Support Vector Machine (SVM) classifier on the training data.
4. **Predict and evaluate** the model on the test data using accuracy or F1-score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIJAY KUMAR D
RegisterNumber: 25000878 
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

## CL (CLASSIFICATION REPORT)

![WhatsApp Image 2025-11-12 at 11 24 10_5a67ace9](https://github.com/user-attachments/assets/df11c15f-5fd1-4316-bdb1-765b5dd1fb89)

## CONFUSION MATRIX

![IMG-20251112-WA0007](https://github.com/user-attachments/assets/3c160b61-0fdf-4e2f-a6d5-a2d2841945bd)

## ACC (ACCURACY)

![IMG-20251112-WA0006](https://github.com/user-attachments/assets/c136bee9-2afb-44a9-856a-6826b9e03928)

## X_TRAIN()

![IMG-20251112-WA0004](https://github.com/user-attachments/assets/1dde81c1-f390-4091-a58e-c0460a43b6d2)

## X.SHAPE()

![IMG-20251112-WA0003](https://github.com/user-attachments/assets/0334a987-2773-47ba-a7c2-f96c76d854a8)

## Y.SHAPE()

![IMG-20251112-WA0002](https://github.com/user-attachments/assets/57aef6c6-cc5d-4020-b0d1-92881e5ae756)

## DATA

![IMG-20251112-WA0001](https://github.com/user-attachments/assets/23e81ba3-7328-4bea-82c0-3fb1872191a6)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
