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

fifa.loc[fifa["prefers_cb"]==True, 'positions'] = "def"
fifa.loc[fifa["prefers_lb"]==True, 'positions'] = "def"
fifa.loc[fifa["prefers_rb"]==True, 'positions'] = "def"
fifa.loc[fifa["prefers_cm"]==True, 'positions'] = "mid"
fifa.loc[fifa["prefers_cam"]==True, 'positions'] = "mid"
fifa.loc[fifa["prefers_cdm"]==True, 'positions'] = "mid"
fifa.loc[fifa["prefers_lm"]==True, 'positions'] = "mid"
fifa.loc[fifa["prefers_rm"]==True, 'positions'] = "mid"
fifa.loc[fifa["prefers_st"]==True, 'positions'] = "att"
fifa.loc[fifa["prefers_lw"]==True, 'positions'] = "att"
fifa.loc[fifa["prefers_rw"]==True, 'positions'] = "att"

print(fifa["positions"])
