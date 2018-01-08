import pandas as pd 
#reads in data
fifa = pd.read_csv("C:/Users/zdunkerton/Documents/FIFA18_Analysis/ingame_positions/mid_ingame.csv")

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
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators = 10, min_samples_split = 10,min_samples_leaf = 5)
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(max_depth=10)
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=10, weights="distance")

from sklearn.ensemble import VotingClassifier
voting_clf=VotingClassifier(estimators=[('rnd',rnd_clf),('dt',dt_clf),('knn',knn_clf)],voting='hard')
voting_clf.fit(features_train, labels_train)
pred = voting_clf.predict(features_test)
print("_______________Soft Vote________________")
print(accuracy_score(labels_test, pred))


dt_clf.fit(features_train, labels_train)
pred = dt_clf.predict(features_test)
print("_______________Decision Tree________________")
print(accuracy_score(labels_test, pred))
#print(precision_score(labels_test, pred, average = 'micro'))
#print(recall_score(labels_test, pred, average = 'micro'))
#print(f1_score(labels_test, pred, average = 'micro'))
#print(classification_report(labels_test, pred))
importance = dt_clf.feature_importances_
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


knn_clf.fit(features_train, labels_train)
pred = knn_clf.predict(features_test)
print("_______________KNN________________")
print(accuracy_score(labels_test, pred))
#print(precision_score(labels_test, pred, average = 'micro'))
#print(recall_score(labels_test, pred, average = 'micro'))
#print(f1_score(labels_test, pred, average = 'micro'))

rnd_clf.fit(features_train, labels_train)
pred = rnd_clf.predict(features_test)
print("_______________Random Forest_______________")
print(accuracy_score(labels_test, pred))
#print(precision_score(labels_test, pred, average = 'micro'))
#print(recall_score(labels_test, pred, average = 'micro'))
#print(f1_score(labels_test, pred, average = 'micro'))

from sklearn.ensemble import BaggingClassifier

#bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=500, bootstrap=True, oob_score=True)
#bag_clf.fit(data, labels)
#print("_______________Bagging________________")
#print("OOB Score: ", bag_clf.oob_score_)

#from sklearn.ensemble import AdaBoostClassifier
#ada_clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=500, algorithm='SAMME.R', learning_rate=.8)
#ada_clf.fit(features_train, labels_train)
#pred = ada_clf.predict(features_test)
#print("_______________AdaBoost Tree_______________")
#print(accuracy_score(labels_test, pred))

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(data)
print(pca.explained_variance_ratio_)

