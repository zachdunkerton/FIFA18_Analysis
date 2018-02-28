#This code adds the position of the players as a new column
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 

fifa = pd.read_csv("./complete.csv")
#create a new column and fill it with random data
position = pd.DataFrame(np.random.randint(low=0, high=5, size=(len(fifa), 1)),columns=['positions'])
#add that column
fifa = fifa.append(position)

#this code adds the position of the player if they are marked for true for that position
cb = fifa.loc[(fifa["prefers_cb"]==True)]
cb["position"] = "cb"
wb = fifa.loc[(fifa["prefers_rb"]==True)|(fifa["prefers_lb"]==True)|(fifa["prefers_rwb"]==True)|(fifa["prefers_lwb"]==True)]
wb["position"] = "wb"
st = fifa.loc[(fifa["prefers_st"]==True)|(fifa["prefers_cf"]==True)]
st['position'] = "st"
wingers = fifa.loc[(fifa["prefers_lm"]==True)|(fifa["prefers_rm"]==True)|(fifa["prefers_lw"]==True)|(fifa["prefers_rw"]==True)]
wingers['position'] = "win"
mid = fifa.loc[(fifa["prefers_cdm"]==True)|(fifa["prefers_cm"]==True)|(fifa["prefers_cam"]==True)]
mid['position'] = "mid"

#combine the data
data = [cb,wb,st,mid, wingers]
result = pd.concat(data)
#sort it by overall rating
result = result.sort_values('overall', ascending=False)
#remove duplicates:some players have more than one prefered position
result = result.drop_duplicates('name')
#put the ingame stats and name into a new data set
result = result.loc[:,["name","club_logo","flag","photo" ,"acceleration","sprint_speed", "positioning","finishing","shot_power","long_shots","volleys","penalties","vision","crossing","free_kick_accuracy","short_passing","long_passing","curve","agility","balance","reactions","ball_control","dribbling","composure","interceptions","heading_accuracy","marking","standing_tackle","sliding_tackle","jumping","stamina","strength","aggression", "position"]]
#need the encoding for the names because they have special characters
result.to_csv("./more_positions_data.csv", encoding = 'utf-8')