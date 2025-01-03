# Titanic-data-analysis

# Titanic Survival Prediction

This project demonstrates how to perform exploratory data analysis (EDA) and build machine learning models to predict survival on the Titanic. The dataset used is `titanic.csv`, which contains information about passengers, such as age, gender, class, and survival status.

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
3. [Running the Code](#running-the-code)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [License](#license)

---

## **Prerequisites**
Before running the code, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  ```bash
  pip install numpy pandas seaborn matplotlib scikit-learn
  ```
- Jupyter Notebook (optional, for running `.ipynb` files).

---

## **Getting Started**
1. **Clone the Repository**  
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Download the Dataset**  
   Ensure the dataset `titanic.csv` is in the same directory as the script or notebook.

---

## **Running the Code**
1. **Using Jupyter Notebook**  
   - Open the `.ipynb` file in Jupyter Notebook.
   - Run each cell sequentially to execute the code.

2. **Using Python Script**  
   - Save the code in a `.py` file (e.g., `titanic_prediction.py`).
   - Run the script using:
     ```bash
     python titanic_prediction.py
     ```

---

## **Code Explanation**
### **1. Import Libraries**
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```
- Libraries used for data manipulation, visualization, and modeling.

### **2. Load and Explore Data**
```python
titanic_data = pd.read_csv('titanic.csv')
titanic_data.head()
titanic_data.shape
titanic_data.info()
titanic_data.isnull().sum()
```
- Load the dataset and explore its structure, summary statistics, and missing values.

### **3. Data Visualization**
```python
sns.countplot(data=titanic_data, x='Survived')
sns.countplot(data=titanic_data, x='Sex', hue='Survived')
sns.countplot(data=titanic_data, x='Pclass', hue='Survived')
sns.histplot(data=titanic_data, x='Age', kde=True)
sns.histplot(data=titanic_data, x='Fare', kde=True)
```
- Visualize the distribution of survival, gender, class, age, and fare.

### **4. Correlation Analysis**
```python
sns.heatmap(data.corr(), annot=True)
print(data.corr())
```
- Analyze correlations between features using a heatmap.

### **5. Data Preprocessing**
```python
data = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
data.dropna(inplace=True)
```
- Drop unnecessary columns, encode categorical variables, and handle missing values.

### **6. Train-Test Split**
```python
from sklearn.model_selection import train_test_split
x = data.drop(columns='Survived')
y = data['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
```
- Split the data into training and testing sets.

### **7. Train Models**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

lg = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
sv = SVC()
gb = GaussianNB()

lg.fit(x_train, y_train)
knn.fit(x_train, y_train)
dt.fit(x_train, y_train)
rf.fit(x_train, y_train)
sv.fit(x_train, y_train)
gb.fit(x_train, y_train)
```
- Train multiple machine learning models.

### **8. Evaluate Models**
```python
from sklearn import metrics

print('Prediction Accuracy')
print('LogisticRegression: ', metrics.accuracy_score(y_test, lg.predict(x_test)))
print('K-Neighbors: ', metrics.accuracy_score(y_test, knn.predict(x_test)))
print('Decision Tree: ', metrics.accuracy_score(y_test, dt.predict(x_test)))
print('Random Forest: ', metrics.accuracy_score(y_test, rf.predict(x_test)))
print('Support Vector: ', metrics.accuracy_score(y_test, sv.predict(x_test)))
print('Gaussian NB: ', metrics.accuracy_score(y_test, gb.predict(x_test)))

print("F1 Score")
print("Logistic: ", metrics.f1_score(y_test, lg.predict(x_test)))
print("K-Neighbors: ", metrics.f1_score(y_test, knn.predict(x_test)))
print("Decision Tree: ", metrics.f1_score(y_test, dt.predict(x_test)))
print("Random Forest: ", metrics.f1_score(y_test, rf.predict(x_test)))
print("Support Vector: ", metrics.f1_score(y_test, sv.predict(x_test)))
print("Gaussian NB: ", metrics.f1_score(y_test, gb.predict(x_test)))
```
- Evaluate models using accuracy and F1 score.

---

## **Results**
- **Best Performing Model**: Logistic Regression achieved the highest accuracy (82.3%) and F1 score (78.2%).
- **Feature Importance**: `Sex`, `Pclass`, and `Fare` were the most influential features in predicting survival.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as needed.

---

## **Support**
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at [minthukywe2020@gmail.com](mailto:minthukywe2020@gmail.com).

---

Enjoy exploring the Titanic survival prediction model! 🚀
