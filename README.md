# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries
2. Load Dataset
3. Drop Unnecessary Data
4. Seperate Features and Target
5. Feature Scaling
6. Train-Test Split
7. Train Logistic Regression
8. Model Prediction
9. Evaluation
10. Confusion Matrix

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())


data = data.drop(["sl_no", "salary"], axis=1)


data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})


X = data.drop("status", axis=1)
y = data["status"]


X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

```

## Output:
<img width="801" height="818" alt="image" src="https://github.com/user-attachments/assets/2f7feffc-6c7f-418a-bf1e-228a92a18477" />
<img width="790" height="810" alt="image" src="https://github.com/user-attachments/assets/0afcc916-5df3-4d0e-93b2-b04905b45743" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
