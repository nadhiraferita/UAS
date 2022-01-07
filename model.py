#library
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

#dataset 
rumah = pd.read_excel('dekoruma.xlsx')

X = np.array(rumah.iloc[:, 1]).reshape((-1, 1))
y = np.array(rumah.iloc[:, 0])

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


#Import library pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


#Fitting model with trainig data
regressor.fit(X.values, y)


import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[99, 82]]))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[99, 82]]))
