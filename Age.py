#this code examines the effect of age on various ratings, as well as get feature importances in a regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor

fifa=pd.read_csv('./complete.csv')
average_pace = []
average_rating = []
ages = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
#for each age, go through the data and get the average pace and overall rating
for age in ages: 
	pace =fifa.loc[fifa["age"]==age,'pac'].mean()
	average_pace.append(pace)

	overall =fifa.loc[fifa["age"]==age,'overall'].mean()
	average_rating.append(overall)

#plot these two on a graph
fig, ax = plt.subplots()
pace = ax.plot(ages, average_pace, '.r-')
rating = ax.plot(ages, average_rating, 'xb-')
ax.legend((pace[0],rating[0]), ('Pace','Rating'))
plt.xlabel("Age")
plt.ylabel("Average Values")
plt.show()

#get the data to run a regression of ingames vs age
data = fifa.loc[:,["acceleration","sprint_speed", "positioning","finishing","shot_power","long_shots","volleys","penalties","vision","crossing","free_kick_accuracy","short_passing","long_passing","curve","agility","balance","reactions","ball_control","dribbling","composure","interceptions","heading_accuracy","marking","standing_tackle","sliding_tackle","jumping","stamina","strength","aggression"]]
labels = fifa["age"]

clf=RandomForestRegressor(n_estimators = 100, max_depth=30, min_samples_leaf=2, min_samples_split=2, oob_score=True)
clf.fit(data, labels)
print(clf.oob_score_)

#graph feature importances
labels=('Acceleration', 'Sprint Speed','Positioning','Finishing','Shot Power','Long Shots','Volleys','Penalties','Vision','Crossing','Free Kicks', 'Short Pass','Long Pass','Curve',"Agility",'Balance','Reactions','Ball Control','Dribbling','Composure','Interceptions','Heading','Marking','Stand Tackle','Slide Tackle','Jumping','Stamina','Strength','Aggression')
pos = np.arange(len(labels))
plt.bar(pos, clf.feature_importances_, align='center', alpha=0.5, color = 'Blue')
ax.set_ylabel('Importance')
ax.set_title('Importance of ingame stats in Age regression')
plt.xticks(pos, labels,rotation=90)
plt.show()