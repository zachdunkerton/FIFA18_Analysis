import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

fifa=pd.read_csv('./complete.csv')
average_pace = []
average_rating = []
ages = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
for age in ages: 
	pace =fifa.loc[fifa["age"]==age,'pac'].mean()
	average_pace.append(pace)

	overall =fifa.loc[fifa["age"]==age,'overall'].mean()
	average_rating.append(overall)

fig, ax = plt.subplots()
pace = ax.plot(ages, average_pace, '.r-')
rating = ax.plot(ages, average_rating, 'xb-')
ax.legend((pace[0],rating[0]), ('Pace','Rating'))
plt.xlabel("Age")
plt.ylabel("Average Values")
plt.show()

data = fifa.loc[:,["acceleration","sprint_speed", "positioning","finishing","shot_power","long_shots","volleys","penalties","vision","crossing","free_kick_accuracy","short_passing","long_passing","curve","agility","balance","reactions","ball_control","dribbling","composure","interceptions","heading_accuracy","marking","standing_tackle","sliding_tackle","jumping","stamina","strength","aggression"]]
labels = fifa["age"]
cols = data.columns.values
features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)

clf=RandomForestRegressor(n_estimators = 100, max_depth=30, min_samples_leaf=2, min_samples_split=2)
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
    #print(stuff[i][0], ":", stuff[i][1])
    i+=1

labels=('Acceleration', 'Sprint Speed','Positioning','Finishing','Shot Power','Long Shots','Volleys','Penalties','Vision','Crossing','Free Kicks', 'Short Pass','Long Pass','Curve',"Agility",'Balance','Reactions','Ball Control','Dribbling','Composure','Interceptions','Heading','Marking','Stand Tackle','Slide Tackle','Jumping','Stamina','Strength','Aggression')
pos = np.arange(len(labels))
plt.bar(pos, clf.feature_importances_, align='center', alpha=0.5, color = 'Blue')
ax.set_ylabel('Importance')
ax.set_title('Importance of ingame stats in Age regression')
plt.xticks(pos, labels,rotation=90)
plt.show()