# import libraries
# ================

# for date and time opeations
from datetime import datetime, timedelta
# for file and folder operations
import os
import math
# for regular expression opeations
import re
import sys
# for listing files in a folder
import glob
# for getting web contents
import requests 
# storing and analysing data
import pandas as pd
# for scraping web contents
from bs4 import BeautifulSoup
# to download data
import wget
# numerical analysis
import numpy as np
# XGBoost
import xgboost as xgb
# regression model
from xgboost import XGBRegressor
# for plotting decision tree
from xgboost import plot_tree
# for data segmentation
from sklearn.model_selection import train_test_split
# for testing accuracy
from sklearn.metrics import mean_absolute_error
# for plotting graphs
import seaborn as sns
# for plotting graphs
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D

from dill.source import getname

import copy

plt.style.use('fivethirtyeight')
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# dataset
# ======

c_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
r_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
d_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

c_map = np.zeros((len(c_df.columns) - 4, 180, 360))
r_map = np.zeros((len(r_df.columns) - 4, 180, 360))
d_map = np.zeros((len(d_df.columns) - 4, 180, 360))


maps = ['c_map', 'r_map', 'd_map']

maps_dict = {
    'c_map': c_map,
    'r_map': r_map,
    'd_map': d_map
}

df_dict = {
    'c_map': c_df,
    'r_map': r_df,
    'd_map': d_df
}


for map in maps:
    try:
        # to reduce processing time, if the .bin has already been created load it
        temp = map + ".bin"
        f = open(temp,"rb")
        maps_dict[map] = np.load(f)
        f.close()

    except IOError:
        temp = np.zeros((len(df_dict[map].columns) - 4, 180, 360))
        # else generate 3D matrix from dataset
        for row in range(0, len(df_dict[map])):
            print(map, "matrix generation\n", "Row ", row,"/", len(df_dict[map]), "\n")
            for date in range(4, len(df_dict[map].columns)):
            # disregards inputs that are nan
                if not (np.isnan(df_dict[map].iloc[row][2]) or np.isnan(df_dict[map].iloc[row][3]) or np.isnan(df_dict[map].iloc[row][date])):
                    #print(df_dict[map].iloc[row][date])
                    maps_dict[map][date - 4, math.floor(df_dict[map].iloc[row][2])+90, math.floor(df_dict[map].iloc[row][3])+180] = df_dict[map].iloc[row][date]
        # generate number of confirmed per day as opposed to total confirmed
        
        for date in range(0, len(df_dict[map].columns) - 4):
            print(map, "difference calculation\n", "Date ", date,"/", len(df_dict[map].columns) - 4, "\n")
            for row in range(0, 179):
                for col in range(0, 359):
                    if date == 0: 
                        # saves the first day
                        temp[date, row, col] = maps_dict[map][date, row, col]
                    else:
                        # current day = current day - previous day
                        temp[date, row, col] = maps_dict[map][date, row, col] - maps_dict[map][date - 1, row, col]
        maps_dict[map] = copy.deepcopy(temp)
        # save the processed matrix into x_xxx.bin
        f = open(map + ".bin", "wb")
        np.save(f, maps_dict[map])
        f.close() 

"""
fig = plt.figure()
z,x,y = maps_dict['c_map'].nonzero()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, -z, zdir='z', c= 'red')

plt.show()
print(c_map)
"""


#-------TESTING NEW METHOD WITH DMatrix-------
#---------------------------------------------


print("c_map shape: ", c_map.shape)
print("r_map shape: ", r_map.shape)
print("d_map shape: ", d_map.shape)

X = np.zeros((3, len(maps_dict['c_map'].flatten())))
y = []

X[0] = maps_dict['c_map'].flatten() 
X[1] = maps_dict['r_map'].flatten() 
X[2] = maps_dict['d_map'].flatten() 

y = copy.deepcopy(X)
X = X.transpose()
y = y.transpose()

dim1, dim2 = X.shape

y = y.flatten()

l = y.shape

y = y[l[0] - dim1:]

print("X shape: ", X.shape)
print("y shape: ", y.shape)

# dividing data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.5)
# instantiating regression model


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_valid, label=y_valid)

print("X_train shape: ", X_train.shape)
print("X_valid shape: ", X_valid.shape)
print("y_train shape: ", y_train.shape)
print("y_valid shape: ", y_valid.shape)

""
try:
    # code for loading the model

	X_full = xgb.DMatrix(X, label=y)

	loaded_model = xgb.Booster()
	loaded_model.load_model("xgboost_covid_optimized.model")

	prediction = loaded_model.predict(X_full)

	print("prediction shape: ", prediction.shape)

	output = prediction.reshape((-1, 180, 360))

	print("output shape: ", output.shape)
	"""
	fig = plt.figure()
	z,x,y = output.nonzero()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, -z, zdir='z', c= 'red')

	plt.show()
	"""

	for row in range(0, 180):
		for col in range(0, 360):
			if output[434][row][col] < 1:
				output[434][row][col] = 0

	np.set_printoptions(threshold=sys.maxsize)
	print(output[434])

	temp = output[434]

	#plt.scatter(temp[:,0], temp[:,1])m,n = a.shape
	m,n = output[434].shape
	R,C = np.mgrid[:m,:n]
	out = np.column_stack((C.ravel(),R.ravel(), output[434].ravel()))
	X = out[:,0]
	Y = out[:,1]
	Z = out[:,2]


	r = 10  # in units of sq pixels

	fig, ax = plt.subplots()
	sc = ax.scatter(X, Y, s=Z, alpha=0.25, edgecolors='blue')

	#plt.imshow(output[434], interpolation='none', origin='lower', cmap=cmap)
	plt.show()



except IOError:

	covid_model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

	# fitting regression model with early_stopping_rounds, which tests model fit with increasing
	# n_estimators and stops when there are 5 consecutive increases in MAE
	covid_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

	print("X_valid", X_valid)
	print(type(X_valid))
	#X_valid.sort_index(inplace=True)
	print("X_valid_sorted", X_valid)
	print("X", X)

	# outputs prediction values for the set
	predictions = covid_model.predict(X_valid)

	# generates MAE value
	print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

	# determining baseline for MAE 
	mean_train = np.mean(y_train)

	# gets predictions on the test set
	baseline_predictions = np.ones(y_valid.shape) * mean_train

	# computes MAE baseline
	mae_baseline = mean_absolute_error(y_valid, baseline_predictions)

	print("Baseline MAE is {:.2f}".format(mae_baseline))

	# managing parameters for model
	params = {
	    # Parameters that we are going to tune.
	    'max_depth':11,
	    'min_child_weight': 1,
	    'eta':.01,
	    'subsample': 1.0,
	    'colsample_bytree': 0.9,
	    # Other parameters
	    'objective':'reg:squarederror',
	    'eval_metric': "mae",
	}

	# max number
	num_boost_round = 999

	# secondary covid model using dmatrix
	covid_model_dm = xgb.train(
		params, 
		dtrain, 
		num_boost_round=num_boost_round, 
		evals=[(dtest, "Test")], 
		early_stopping_rounds=10
	)

	# secondary covid results with different approach
	dm_cv_results = xgb.cv(
		params,
		dtrain,
		num_boost_round=num_boost_round,
		seed=42,
		nfold=5,
		metrics={'mae'},
		early_stopping_rounds=10
	)

	print(dm_cv_results['test-mae-mean'].min())

	# these are ranges that can be adjusted

	gridsearch_params = [
	    (max_depth, min_child_weight)
	    for max_depth in range(9,12)
	    for min_child_weight in range(1,8)
	]


	# not necessaryto run cross validation each time
	# best params found are max_depth=11, min_child_weight=1

	# define initial best params and MAE
	min_mae = float("Inf")
	best_params = None
	for max_depth, min_child_weight in gridsearch_params:
	    print("CV with max_depth={}, min_child_weight={}".format(
	                             max_depth,
	                             min_child_weight))

	    # Update our parameters
	    params['max_depth'] = max_depth
	    params['min_child_weight'] = min_child_weight

	    # Run CV
	    cv_results = xgb.cv(
	        params,
	        dtrain,
	        num_boost_round=num_boost_round,
	        seed=42,
	        nfold=5,
	        metrics={'mae'},
	        early_stopping_rounds=10
	    )

	    # Update best MAE
	    mean_mae = cv_results['test-mae-mean'].min()
	    boost_rounds = cv_results['test-mae-mean'].argmin()
	    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	    if mean_mae < min_mae:
	        min_mae = mean_mae
	        best_params = (max_depth,min_child_weight)

	print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


	# best settings as output by tests above
	params['max_depth']=best_params[0]
	params['min_child_weight']=best_params[1]

	# best params subsample=1.0, colsample_bytree=0.9

	gridsearch_params = [
	    (subsample, colsample)
	    for subsample in [i/10. for i in range(7,11)]
	    for colsample in [i/10. for i in range(7,11)]
	]

	min_mae = float("Inf")
	best_params = None
	# We start by the largest values and go down to the smallest
	for subsample, colsample in reversed(gridsearch_params):
	    print("CV with subsample={}, colsample={}".format(
	                             subsample,
	                             colsample))
	    # We update our parameters
	    params['subsample'] = subsample
	    params['colsample_bytree'] = colsample
	    # Run CV
	    cv_results = xgb.cv(
	        params,
	        dtrain,
	        num_boost_round=num_boost_round,
	        seed=42,
	        nfold=5,
	        metrics={'mae'},
	        early_stopping_rounds=10
	    )
	    # Update best score
	    mean_mae = cv_results['test-mae-mean'].min()
	    boost_rounds = cv_results['test-mae-mean'].argmin()
	    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
	    if mean_mae < min_mae:
	        min_mae = mean_mae
	        best_params = (subsample,colsample)

	print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))



	# best settings as output by tests above
	params['subsample']=best_params[0]
	params['colsample_bytree']=best_params[1]

	# eta=0.01

	# This can take some time…
	min_mae = float("Inf")
	best_params = None
	for eta in [.3, .2, .1, .05, .01, .005]:
	    print("CV with eta={}".format(eta))
	    # We update our parameters
	    params['eta'] = eta
	    # Run and time CV
	    cv_results = xgb.cv(
	            params,
	            dtrain,
	            num_boost_round=num_boost_round,
	            seed=42,
	            nfold=5,
	            metrics=['mae'],
	            early_stopping_rounds=10
	          )
	    # Update best score
	    mean_mae = cv_results['test-mae-mean'].min()
	    boost_rounds = cv_results['test-mae-mean'].argmin()
	    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
	    if mean_mae < min_mae:
	        min_mae = mean_mae
	        best_params = eta
	print("Best params: {}, MAE: {}".format(best_params, min_mae))

	# best settings as output by tests above
	params['eta']=best_params

	num_boost_round = covid_model_dm.best_iteration + 1

	optimized_covid_model_dm = xgb.train(
		params,
		dtrain,
		num_boost_round=num_boost_round,
		evals=[(dtest, "Test")]
	)

	print(mean_absolute_error(optimized_covid_model_dm.predict(dtest), y_valid))

	print(optimized_covid_model_dm.predict(dtest))

	optimized_covid_model_dm.save_model("xgboost_covid_optimized.model")

	plot_tree(optimized_covid_model_dm)

	plt.show()


