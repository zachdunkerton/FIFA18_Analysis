#This code is a regression analysis of the ratings of players based on their position stats. 

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_absolute_error	

#read in the whole data set
fifa = pd.read_csv("./complete.csv")
#labels are the overall rating of the player
labels = fifa["overall"]
#choose the sapecific columns for the in game stats from the complete dataset
data=fifa.loc[:,["acceleration","sprint_speed", "positioning","finishing","shot_power","long_shots","volleys","penalties","vision","crossing","free_kick_accuracy","short_passing","long_passing","curve","agility","balance","reactions","ball_control","dribbling","composure","interceptions","heading_accuracy","marking","standing_tackle","sliding_tackle","jumping","stamina","strength","aggression"]]
cols = data.columns.values
#train-test split the data
features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)
#parameters used in the gridsearch to optimize the algorithm 
parameters = {'max_depth':[2,5,10,100], 'min_samples_split':[2,5,10,100], 'min_samples_leaf':[1,2,5,10]}
clf = DecisionTreeRegressor(max_depth=100, min_samples_leaf=10, min_samples_split=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(mean_absolute_error(labels_test, pred)) 

from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators = 10, max_depth=30, min_samples_leaf=2, min_samples_split=2, criterion = 'mae')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(mean_absolute_error(labels_test, pred))
#print the feature importances of the algorithm
importance = clf.feature_importances_
stuff = []
i= 0
while i < len(cols):
    stuff.append([cols[i], importance[i]])
    i+=1
#Sorts them so they appear greatest to least
stuff = sorted(stuff,key=lambda x: (x[1]), reverse = True)
i= 0
while i < len(stuff):
    print(stuff[i][0], ":", stuff[i][1])
    i+=1