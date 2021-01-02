# Chocolate Rating

Data Preprocessing, Exploratory Data Analysis and Machine Learning to predict low and high rated chocolate bars. 

The dataset is available in Kaggle at:

[https://www.kaggle.com/rtatman/chocolate-bar-ratings](https://www.kaggle.com/rtatman/chocolate-bar-ratings)

## Libraries


```python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
import plotly as pl 
import plotly.graph_objs as gobj
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
from imblearn.under_sampling import ClusterCentroids
init_notebook_mode(connected=True)
```
The last line initializes the plotly map in jupyter notebook instance. 


## Chocolate Rating based on Bean Origin

<p align="center">
  <img src="https://github.com/haxamxam/Chocolate_Rating/blob/master/bean_origin_rating.png" title="Rating Bean Origin">
</p>

## Chocolate Rating based on Company Location

<p align="center">
  <img src="https://github.com/haxamxam/Chocolate_Rating/blob/master/company_location_rating.png" title="Rating Company Location">
</p>
