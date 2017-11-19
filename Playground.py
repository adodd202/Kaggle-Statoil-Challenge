#Reference code: kernel - exploration-transforming-images-in-python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
#matplotlib inline

#with open('/Users/adodd202/Documents/GitHub/Kaggle-Statoil-Challenge/Data/train.json') as f:
#   train = json.load(f)

train = pd.read_json('/Users/adodd202/Documents/GitHub/Kaggle-Statoil-Challenge/Data/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors = 'coerce')

print (train.shape)

test = pd.read_json('/Users/adodd202/Documents/GitHub/Kaggle-Statoil-Challenge/Data/test.json')
#test['inc_angle'] = pd.to_numeric(train['inc_angle'], errors = 'coerce')

print (test.shape)