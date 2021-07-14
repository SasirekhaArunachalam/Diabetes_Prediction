# Import Libraries and Dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import warnings
warnings.filterwarnings('ignore')

#Import Dataset 

dataset = pd.read_csv('diabetes.csv')

# DESCRIPTIVE STATICSTICS 
# Preview data
print("Preview data")
print(dataset.head())

# Dataset dimensions
print("Dataset dimensions")
print(dataset.shape)

# Features data - type
print("Features data - type")
print(dataset.info())

# Statistical Summary 
print("Statistical Summary ")
print(dataset.describe().T)

# Count of all null values 
print("Count of all null values ")
dataset.isnull().sum()

'''## Observations:
1. There are a total of 768 records and 9 features in the dataset.
2. Each feature can be either of integer or float dataype.
3. Some features like Glucose, Blood pressure , Insulin, BMI have zero values which represent missing data.
4. There are zero NaN values in the dataset.
5. In the outcome column, 1 represents diabetes positive and 0 represents diabetes negative.'''

# DATA VISUALIZATION 

sns.countplot( x = 'Outcome' , data = dataset )
plt.show()


# Histogram of each feature 

import itertools

col = dataset.columns[:8]
plt.subplots(figsize = (20, 15))
length = len(col)

for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length/2), 3, j + 1)
    plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
    dataset[i].hist(bins = 20)
    plt.title(i)
plt.show()




# Pairplot 
sns.pairplot(data = dataset, hue = 'Outcome')
plt.show()

# Heatmap
sns.heatmap(dataset.corr(), annot = True)
plt.show()

'''Observations:
The countplot tells us that the dataset is imbalanced,
 as number of patients who don't have diabetes is more than those who do.
From the correaltion heatmap, we can see that there is a high
 correlation between Outcome and [Glucose,BMI,Age,Insulin].
  We can select these features to accept input from the user and predict the
   outcome.'''


# DATA PREPROCESSING 

dataset_new = dataset

# Replace zero with NaN

dataset_new[["Pregnancies" , "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI" , "DiabetesPedigreeFunction", "Age"]] = dataset_new[["Pregnancies","Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].replace(0, np.NaN) 



# Replacing NaN with mean values
dataset_new["Pregnancies"].fillna(dataset_new["Pregnancies"].mean(), inplace = True)
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)
dataset_new["DiabetesPedigreeFunction"].fillna(dataset_new["DiabetesPedigreeFunction"].mean(), inplace = True)
dataset_new["Age"].fillna(dataset_new["Age"].mean(), inplace = True)

# Statistical summary
print("Statistical summary")
print(dataset_new.describe().T)

# Feature scaling using Min max scaler 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_new)

dataset_scaled = pd.DataFrame(dataset_scaled)

# Selecting features - [Glucose, Insulin, BMI, Age]
X = dataset_scaled.iloc[:, [1, 2 , 4 , 5 , 7]].values
Y = dataset_scaled.iloc[:, 8].values

# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

# Checking dimensions

print("Checking dimensions\n")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# DATA MODELLING


# Logistic Regression Algorithm 

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train, Y_train)

# K nearest neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)

# Support Vector Classifier Algorithm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)

# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)

# Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dectree.fit(X_train, Y_train)

# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, Y_train)

# Making predictions on test dataset
Y_pred_logreg = logreg.predict(X_test)
Y_pred_knn = knn.predict(X_test)
Y_pred_svc = svc.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


# MODEL EVALUATION 

# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score
accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)

# Accuracy on test set
print("Logistic Regression: " + str(accuracy_logreg * 100))
print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_knn)
cm

# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_knn))

pickle.dump(ranfor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
