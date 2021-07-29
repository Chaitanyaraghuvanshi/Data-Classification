# Data-Classification
While the highest accuracy is of the K Nearest Neighbors Algorithm (k=7), the Decision Tree Classifier performed better by 0.009 when compared to the average of the two KNN (k=3, k=7). However, the difference is extremely low indicating there is not much difference between the two. Furthermore, the accuracy of k-nn when k=7 (0.727) performed better than when k=3 (0.673). This can be explained as an algorithm with smaller k value is more likely to be affected by noise than an algorithm with higher k value.
 
The dataset to be split is created by merging and sorting the two csv files, ‘world.csv’ and ‘life.csv’ on the basis of the country code. 
A DataFrame containing the median, mean and variance of each feature is created which is then stored as ‘task2a.csv’. 
The data is split into training and testing dataset with a random state of 200
The missing values are imputed by calculating median of the features in training dataset. Further the values are scaled using the StandardScaler() function. By removing mean and scaling to unit variance.
Each of the three models are fit using the training values and based on it the prediction values are calculated by passing the testing dataset into the fitted model.  
As the output of the code, the accuracies for KNN and DecisionTree Classifiers are shown as follows:
Accuracy of decision tree: 0.709
Accuracy of k-nn (k=3): 0.673
Accuracy of k-nn (k=7): 0.727

Feature Engineering and Selection

Steps Taken:
After merging the dataset and sorting values on the basis of country code, the columns containing the country name, code and year are dropped. Further, the data is split into training and testing dataset for features and class labels. 
The KNN classification is performed, returning a classifier object with k=3. 
The data is imputed using the mean value of the training dataset. The imputed data is then placed at the missing values of the training as well as the testing dataset.
Interaction term pairs are created using the PolynomialFeatures from the preprocessing module which generates interaction features. 
The pre-processing of the engineered data takes place using StandardScaler() function which subtracts the mean and scales data to unit variance. 
The KMeans clustering of the original 20 feature data is performed to get the last feature of our feature engineering model. With the help of this KMeans model we predict the data for our testing dataset of our feature model. This feature is then added to our existing set of 210 features to make the final feature engineering model with 211 features.
Next we determine the four most suitable features that are to be used for performing 3NN classification. This feature selection is done by mutual information with the help of the module, ‘feature_selection’ provided by the sklearn library. Feature selection not only is the accuracy improved but also reduces training time.
The predicted value is calculated using the KNN model fitted with the training dataset. The accuracy of the prediction is then made by our best 4 features from the feature engineering model.  
Next, in order to perform feature engineering, an alternative method of feature engineering is used, called Principal Component Analysis (PCA) which uses the ‘PCA’ function provided by decomposition module in sklearn. This function is used for matrix decomposition to get maximum variance out of the complete dataset.
Lastly, the 3NN classification is performed using the first four features out of the original dataset. The training and testing dataset are imputed by fitting the mean values of the training dataset and similarly scaling is performed to ensure normalisation of data.

Selection of number of clusters
In order to determine the number of clusters, the elbow curve method is used. For this a graph of a range of values (1 to 10) for the number of clusters is plotted against their respective sum of squares distance of the KMeans model as shown in graph 1. The correct number of clusters to use is determined by looking for an ‘elbow’ or bend in the graph which can be observed at a value of 2 clusters in the graph.

Selection of best 4 features
The method used to select the best four features out of our 211 features model is determined by mutual_info_class imported from the feature_selection module of sklearn. Mutual information is not affected by the scaling of data and is robust towards noise. Furthermore, Mutual information is preferred over other methods such as chi-square feature selection, as chi-square makes it difficult to interpret results when there are a large number of features, in our case 211 features. Furthermore, all values should be positive for chi2 which is not in the case of our feature values.

Analysis of the best method 
The feature engineering model containing 211 features produced the best results as the accuracy is higher than the other two methods used as shown below.
Accuracy of feature engineering: 0.709
Accuracy of PCA: 0.673
Accuracy of first four features: 0.636
The accuracy for this method can be accounted by the high variation in our dataset by the use of best four features out of the 211 features calculated. Since the unnecessary data is removed by feature selection, the noise is getting reduced. Furthermore, only 4 features are being used to predict the accuracy which means the algorithm becomes fast.

Techniques to improve classification accuracy
Choosing the best suitable random state can lead to an increase in classification accuracy. For this, the model should be tested on a range of random states using a loop and the one giving best accuracy should be chosen.
Ensemble methods such as Bagging and Boosting can be used in order to increase the classification accuracy. In such methods, multiple models can be integrated in order to reduce error rate and increase prediction accuracy.
Another technique that can be used is k-fold cross validation in which the data is split into k groups wherein each fold is used as the testing data and remaining as training data which helps in reducing bias.

Reliability of classification model
The model is said to be reliable as the data is split into training and testing dataset in a suitable ratio of 0.7:0.3 and the model is fitted only on the training set.
