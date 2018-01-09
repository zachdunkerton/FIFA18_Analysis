import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 

fifa = pd.read_csv("./position_stats.csv")
position = pd.DataFrame(np.random.randint(low=0, high=5, size=(len(fifa), 1)),columns=['positions'])
fifa = fifa.append(position)
cols = fifa.columns.values
#data = pd.DataFrame()
#data.append()
defd = fifa.loc[(fifa["prefers_cb"]==True)|(fifa["prefers_lb"]==True)|(fifa["prefers_rb"]==True)]
defd["position"] = "def"
mid = fifa.loc[(fifa["prefers_cm"]==True)|(fifa["prefers_cdm"]==True)|(fifa["prefers_cam"]==True)|(fifa["prefers_lm"]==True)|(fifa["prefers_rm"]==True)]
mid['position'] = "mid"
att = fifa.loc[(fifa["prefers_st"]==True)|(fifa["prefers_lw"]==True)|(fifa["prefers_rw"]==True)]
att['position'] = "att"

data = [defd, mid, att]
result = pd.concat(data)
result.to_csv("./position_data.csv")