import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
import numpy as np
import itertools
##load in the data
df1=pd.read_csv('world.csv')
df2=pd.read_csv('life.csv')
df= pd.merge(df1, df2, on='Country Code')
df = df.sort_values(by = 'Country Code')
df = df.drop(['Country Code','Country Name','Time','Year','Country'], axis=1)

data=df.iloc[:,:-1]
data= data.replace('..',np.nan)
data = data.astype(float)
classlabel=df['Life expectancy at birth (years)']

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.70, test_size=0.30, random_state=100)

knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)

#imputation of our original 20 feature dataset
imp = SimpleImputer().fit(X_train)
X_train=imp.transform(X_train)
X_test=imp.transform(X_test)

#Getting the interaction pairs for our feature engineering model
poly = preprocessing.PolynomialFeatures(interaction_only = True, include_bias = False)
poly.fit(X_train)
X_train_fe = poly.transform(X_train)
X_test_fe  = poly.transform(X_test)
print('Interaction term pairs (training dataset) :')
print(pd.DataFrame(X_train_fe).iloc[:,20:])
print()
#Scaling the 210 feature model  
scaler = preprocessing.StandardScaler().fit(X_train_fe)
X_train_fe =scaler.transform(X_train_fe)
X_test_fe =scaler.transform(X_test_fe)

#Scaling the original 20 feature dtaset
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

X_train_fe=pd.DataFrame(X_train_fe)
X_test_fe=pd.DataFrame(X_test_fe)

#Using elbow curve method to determine number of clusters
k_range=range(1,10)
ssd=[]
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train)
    ssd.append(kmeans.inertia_)

plt.plot(k_range, ssd)
plt.xlabel('Number of clusters')
plt.ylabel('SSD')
plt.title('Elbow curve to determine no. of clusters')
plt.savefig('task2bgraph1.png', bbox_inches='tight')

#KMeans clustering model
kmeans_model = KMeans(n_clusters=3).fit(X_train)
f_cluster = kmeans_model.labels_
f_cluster_test = kmeans_model.predict(X_test)

print('Clustering feature (training dataset):')
print(f_cluster)
print()

#211 feature datset created by adding the last cluster feature
X_train_fe = pd.concat([X_train_fe,pd.Series(f_cluster)], axis = 1) 
X_test_fe = pd.concat([X_test_fe,pd.Series(f_cluster_test)], axis = 1)

#Selecting best 4 out of the 211 features of our model
selection = SelectKBest(score_func=mutual_info_classif, k=4)
selection.fit(X_train_fe,y_train)
X_train_fe = selection.transform(X_train_fe)
X_test_fe = selection.transform(X_test_fe)

print('Best 4 features selected (training dataset):')
print(pd.DataFrame(X_train_fe))
print()

knn3.fit(X_train_fe ,y_train)
y_pred_feature=knn3.predict(X_test_fe)
acc_fea = accuracy_score(y_test, y_pred_feature)


#PCA feature engineering
pca_features = PCA(n_components = 4)
pca_features.fit(X_train)
X_train_pca = pca_features.transform(X_train)
X_test_pca = pca_features.transform(X_test)

print('PCA features (training dataset):')
print(pd.DataFrame(X_train_pca))
print()

knn3.fit(X_train_pca, y_train)
y_pred_pca=knn3.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)


#first four Features
data=df.iloc[:,:4]
data= data.replace('..',np.nan)
data = data.astype(float)
classlabel=df['Life expectancy at birth (years)']

#Splitting the new dataset of four features
X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.70, test_size=0.30, random_state=100)

#Imputation and Scaling
imp = SimpleImputer(strategy='median').fit(X_train)
X_train=imp.transform(X_train)
X_test=imp.transform(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

print('First 4 features (training dataset):')
print(pd.DataFrame(X_train))
print()

knn3.fit(X_train, y_train)
y_pred_1st4=knn3.predict(X_test)
acc_1st4 = accuracy_score(y_test, y_pred_1st4)

print('Accuracy of feature engineering:',round(acc_fea,3))
print('Accuracy of PCA:',round(acc_pca,3))
print('Accuracy of first four features:',round(acc_1st4,3))
