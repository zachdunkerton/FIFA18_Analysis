#This code classifies players by position based on in-game stats. 
#the data is first read in, then passed through a Random Forest and KNN classifier 
#information is extracted from each of these anlgorithms

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#reads in preprocessed
fifa = pd.read_csv("./more_positions_data.csv")
data = fifa.iloc[:, 2:31]
cols = data.columns.values

#train test split the data
#features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)

#used this to determine best parameters for the Random Forest Trees
#from sklearn.tree import DecisionTreeClassifier
#parameters = {'criterion':['gini','entropy'], 'max_depth':[3,7,12], 'min_samples_split':[2,5,10], 'min_samples_leaf':[1,3,5,10]}
#dt_clf=DecisionTreeClassifier(max_depth=7)
#dt_clf=GridSearchCV(DecisionTreeClassifier(), parameters)
#dt_clf = DecisionTreeClassifier(max_depth=7, criterion='entropy',min_samples_split = 10,min_samples_leaf = 5,)
#dt_clf.fit(features_train, labels_train)
#pred = dt_clf.predict(features_test)
#print("_______________Tree_______________")
#print(accuracy_score(labels_test, pred))
#print(dt_clf.best_params_)
#print(dt_clf.best_score_)

#This is t he Random Tree Classifier
#useful because it yields Feature Weights 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
rnd_clf = RandomForestClassifier(n_estimators = 250, min_samples_split = 10,min_samples_leaf = 5, max_depth=7, oob_score = True)
rnd_clf.fit(data, fifa["position"])
print("_______________Random Forest_______________")
print(rnd.clf.oob_score_)
features = []
#print out the feature importances
i= 0
while i < len(cols):
    features.append([cols[i], rnd_clf.feature_importances_[i]])
    i+=1
#sort
features = sorted(features,key=lambda x: (x[1]), reverse = True)
for x in features:
    print(x)

#This classifier performed the best in terms of accuracy, used here determine similar players
from sklearn.neighbors import KNeighborsClassifier
#parameters = {'n_neighbors':[70,75,80,90]}
#knn_clf=GridSearchCV(KNeighborsClassifier(), parameters)
knn_clf=KNeighborsClassifier(n_neighbors=80, weights ='uniform')
knn_clf.fit(data, fifa["position"])
#the first number is the index of the player you want to get the neighbors of
#number of neighbors can also be changed
n = knn_clf.kneighbors([data.iloc[0,:]],n_neighbors=15)[1]
#the method returns the index of the similar players, in order of most closely related
for i in n:
	print(fifa.iloc[i,[1]])
#pred = knn_clf.predict(features_test)
#print("_______________KNN_______________")
#print(accuracy_score(labels_test, pred))


