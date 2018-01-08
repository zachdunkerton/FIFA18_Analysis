import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_absolute_error	
 
fifa = pd.read_csv("./position_stats.csv")
labels = fifa["overall"]
reactions = fifa["composure"]
plt.scatter(reactions, labels)
plt.show()
data=fifa.iloc[0:len(fifa), 11:40]
cols = data.columns.values
 
features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)
parameters = {'max_depth':[2,5,10,100], 'min_samples_split':[2,5,10,100], 'min_samples_leaf':[1,2,5,10]}
clf = DecisionTreeRegressor(max_depth=100, min_samples_leaf=10, min_samples_split=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
#print(mean_absolute_error(labels_test, pred)) 

from sklearn.linear_model import ElasticNet
parameters = {'alpha':[1,2,5,10], 'l1_ratio':[.1, .15,.2,.25,.3]}
#clf =GridSearchCV(ElasticNet(), parameters)
clf=ElasticNet(alpha=1, l1_ratio=.1)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
diff = np.subtract(labels_test, pred)
diff = np.absolute(diff)
print(np.average(diff))
#print(clf.best_params_)
#print(mean_absolute_error(labels_test, pred))
#plt.figure()
#plt.hist(diff, bins=[0,2,4,6])
#plt.show()


from sklearn.ensemble import RandomForestRegressor

clf=RandomForestRegressor(max_depth=100, min_samples_leaf=10, min_samples_split=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
diff = np.subtract(labels_test, pred)
diff = np.absolute(diff)
print(np.average(diff))
i=0
importance = clf.feature_importances_
stuff = []
i= 0
while i < len(cols):
    stuff.append([cols[i], importance[i]])
    i+=1
stuff = sorted(stuff,key=lambda x: (x[1]), reverse = True)
i= 0
while i < len(stuff):
    print(stuff[i][0], ":", stuff[i][1])
    i+=1