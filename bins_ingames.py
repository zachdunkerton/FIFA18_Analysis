import pandas as pd 
#reads in data
fifa = pd.read_csv("C:/Users/Zach Dunkerton/Documents/Towson/Fall 2017/COSC 757/Final Project/ingame positions/top_ingame.csv")

labels = fifa["overall"]
#ten bins
#i = 0
#while (i < len(fifa)):
#    if (int(fifa.loc[i,'overall'] ) >= 45 and int(fifa.loc[i,'overall'] ) <= 49):
#        fifa.loc[i, 'overall'] = 1
#    if (int(fifa.loc[i,'overall'] ) >= 50 and int(fifa.loc[i,'overall'] ) <= 54):
#        fifa.loc[i, 'overall'] = 2
#    if (int(fifa.loc[i,'overall'] ) >= 55 and int(fifa.loc[i,'overall'] ) <= 59):
#        fifa.loc[i, 'overall'] = 3
#    if (int(fifa.loc[i,'overall'] ) >= 60 and int(fifa.loc[i,'overall'] ) <= 64):
#        fifa.loc[i, 'overall'] = 4
#    if (int(fifa.loc[i,'overall'] ) >= 65 and int(fifa.loc[i,'overall'] ) <= 69):
#        fifa.loc[i, 'overall'] = 5
#    if (int(fifa.loc[i,'overall'] ) >= 70 and int(fifa.loc[i,'overall'] ) <= 74):
#        fifa.loc[i, 'overall'] = 6
#    if (int(fifa.loc[i,'overall'] ) >= 75 and int(fifa.loc[i,'overall'] ) <= 79):
#        fifa.loc[i, 'overall'] = 7
#    if (int(fifa.loc[i,'overall'] ) >= 80 and int(fifa.loc[i,'overall'] ) <= 84):
#        fifa.loc[i, 'overall'] = 8
#    if (int(fifa.loc[i,'overall'] ) >= 85 and int(fifa.loc[i,'overall'] ) <= 89):
#        fifa.loc[i, 'overall'] = 9
#    if (int(fifa.loc[i,'overall'] ) >= 90 and int(fifa.loc[i,'overall'] ) <= 94):
#        fifa.loc[i, 'overall'] = 10
#    i = i + 1

#five bins
i = 0
while (i < len(fifa)):
    if (int(fifa.loc[i,'overall'] ) >= 40 and int(fifa.loc[i,'overall'] ) <= 49):
        fifa.loc[i, 'overall'] = 1
    if (int(fifa.loc[i,'overall'] ) >= 50 and int(fifa.loc[i,'overall'] ) <= 59):
        fifa.loc[i, 'overall'] = 2
    if (int(fifa.loc[i,'overall'] ) >= 60 and int(fifa.loc[i,'overall'] ) <= 69):
        fifa.loc[i, 'overall'] = 3
    if (int(fifa.loc[i,'overall'] ) >= 70 and int(fifa.loc[i,'overall'] ) <= 79):
        fifa.loc[i, 'overall'] = 4
    if (int(fifa.loc[i,'overall'] ) >= 80 and int(fifa.loc[i,'overall'] ) <= 89):
        fifa.loc[i,'overall'] = 5
    if (int(fifa.loc[i, 'overall']) >= 90 and int(fifa.loc[i, 'overall']) <= 99):
            fifa.loc[i, 'overall'] = 6
    i = i + 1


#ID = fifa["ID"]
#need to change this
data = fifa.iloc[0:len(fifa), 3:36]
cols = data.columns.values
print(cols)
print(len(fifa))
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("_______________Decision Tree________________")
print(accuracy_score(labels_test, pred))
#print(precision_score(labels_test, pred, average = 'micro'))
#print(recall_score(labels_test, pred, average = 'micro'))
#print(f1_score(labels_test, pred, average = 'micro'))
#print(classification_report(labels_test, pred))
importance = clf.feature_importances_
stuff = []
i= 0
while i < len(cols):
    stuff.append([cols[i], importance[i]])
    i+=1
stuff = sorted(stuff,key=lambda x: (x[1]), reverse = True)
i= 0
while i < len(stuff):
    #print(stuff[i][0], ":", stuff[i][1])
    i+=1

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=10, weights="distance")
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)

#print("_______________KNN________________")
#print(accuracy_score(labels_test, pred))
#print(precision_score(labels_test, pred, average = 'micro'))
#print(recall_score(labels_test, pred, average = 'micro'))
#print(f1_score(labels_test, pred, average = 'micro'))

#from sklearn.svm import SVC
#clf = SVC()
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#print("_______________SVM_______________")
#print(accuracy_score(labels_test, pred))
#print(precision_score(labels_test, pred, average = 'micro'))
#print(recall_score(labels_test, pred, average = 'micro'))
#print(f1_score(labels_test, pred, average = 'micro'))

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 10, min_samples_split = 10,min_samples_leaf = 5)
#clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#print("_______________Random Forest_______________")
#print(accuracy_score(labels_test, pred))
#print(precision_score(labels_test, pred, average = 'micro'))
#print(recall_score(labels_test, pred, average = 'micro'))
#print(f1_score(labels_test, pred, average = 'micro'))