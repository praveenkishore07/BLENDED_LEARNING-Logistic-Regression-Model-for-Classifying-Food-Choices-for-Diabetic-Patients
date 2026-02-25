# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data Import and prepare the dataset to initiate the analysis workflow.

2.Explore Data Examine the data to understand key patterns, distributions, and feature relationships.

3.Select Features Choose the most impactful features to improve model accuracy and reduce complexity.

4.Split Data Partition the dataset into training and testing sets for validation purposes.

5.Scale Features Normalize feature values to maintain consistent scales, ensuring stability during training.

6.Train Model with Hyperparameter Tuning Fit the model to the training data while adjusting hyperparameters to enhance performance.

7.Evaluate Model Assess the model’s accuracy and effectiveness on the testing set using performance metrics.

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: A PRAVEEN KISHORE
RegisterNumber:  212225220074

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('food_items.csv')

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

l2_model = LogisticRegression(
    random_state=123, 
    penalty='l2', 
    multi_class='multinomial', 
    solver='lbfgs', 
    max_iter=1000
)

l2_model.fit(X_train, y_train)
y_pred = l2_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()

*/
```

## Output:
## DATASET AND INFO--
<img width="915" height="613" alt="1" src="https://github.com/user-attachments/assets/a57c11ff-951d-4636-8847-b93a91de4daa" />

<img width="617" height="522" alt="2" src="https://github.com/user-attachments/assets/3c7708e7-1a62-452e-9a73-fdcb068cb543" />

## Performance metrics--
<img width="631" height="280" alt="3" src="https://github.com/user-attachments/assets/029979d6-6b3a-468d-8709-42493d832bb5" />

## Confusion Metrics--
<img width="738" height="535" alt="4" src="https://github.com/user-attachments/assets/9e4f70fb-7556-4ff4-a305-770040799ecf" />
## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
