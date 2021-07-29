import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
##load in the data
df1=pd.read_csv('world.csv')
df2=pd.read_csv('life.csv')
df= pd.merge(df1, df2, on='Country Code')
df = df.sort_values(by = 'Country Code')
df = df.drop(['Country Code','Country Name','Time','Year','Country'], axis=1)

data=df.iloc[:,:-1]
data= data.replace('..',np.nan)
data = data.astype(float)

task2a_df = pd.DataFrame({'feature': data.columns, 'median':round(data.median(), 3),'mean':round(data.mean(), 3), 'variance': round(data.var(),3)})
task2a_df.to_csv(r'task2a.csv',index = False)
classlabel=df['Life expectancy at birth (years)']
X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.70, test_size=0.30, random_state=200)

#imputation and scaling
imp = SimpleImputer(strategy='median').fit(X_train)
X_train=imp.transform(X_train)
X_test=imp.transform(X_test)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Decision tree
dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)
y_pred1=dt.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)
print('Accuracy of decision tree:', round(acc1, 3))

#KNN with k=3
knn1 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train, y_train)
y_pred2=knn1.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
print('Accuracy of k-nn (k=3):', round(acc2, 3))

#KNN with k=7
knn2 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn2.fit(X_train, y_train)
y_pred3=knn2.predict(X_test)
acc3=accuracy_score(y_test, y_pred3)
print('Accuracy of k-nn (k=7):', round(acc3, 3))