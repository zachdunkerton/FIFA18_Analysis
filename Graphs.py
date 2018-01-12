import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

position_fifa = pd.read_csv("./more_positions_data.csv")
position_labels = position_fifa["position"]
position_data = position_fifa.iloc[:, 1:30]

position_rnd_clf = RandomForestClassifier(n_estimators = 250, min_samples_split = 10,min_samples_leaf = 5, max_depth=7)
position_rnd_clf.fit(position_data, position_labels)

overall_fifa=pd.read_csv('./complete.csv')
overall_labels = overall_fifa["overall"]
overall_data = overall_fifa.loc[:,["acceleration","sprint_speed", "positioning","finishing","shot_power","long_shots","volleys","penalties","vision","crossing","free_kick_accuracy","short_passing","long_passing","curve","agility","balance","reactions","ball_control","dribbling","composure","interceptions","heading_accuracy","marking","standing_tackle","sliding_tackle","jumping","stamina","strength","aggression"]]
overall_rnd_clf = RandomForestRegressor(n_estimators = 100, max_depth=30, min_samples_leaf=2, min_samples_split=2)
overall_rnd_clf.fit(overall_data, overall_labels)

width = .4 
fig, ax = plt.subplots()
overall = ax.bar(pos, overall_rnd_clf.feature_importances_, width, align='center', alpha=0.5, color = 'Blue')

position = ax.bar(pos+width, position_rnd_clf.feature_importances_, width, align='center', alpha=0.5, color = 'Red')
ax.set_xticks(pos + width / 2)
ax.set_xticklabels(labels,rotation=90)
ax.set_ylim([0,.2])
ax.legend((overall[0],position[0]), ('Overall','Position'))
ax.set_ylabel('Importance')
ax.set_title('Importance of ingame stats in position classification vs overall classification')
plt.show()
