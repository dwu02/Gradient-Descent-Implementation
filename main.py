import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time as t
import sys
from algorithms import OLS, uniLin
from algorithms import multiLin
from display_results import displayMultivariate, displayUnivariate

start_elapsed = t.time()
if len(sys.argv) > 12 or len(sys.argv) < 12: 
    print('Supply a correct number command line arguments')
    sys.exit()

file = sys.argv[1]
training_range =  int(sys.argv[2])
regression_type =  sys.argv[3]
predictor_vars = [int(var) for var in sys.argv[4].split(',')]
response_var = int(sys.argv[5])
learning_rates = [float(i) for i in sys.argv[6].split(',')]
epoch_max = int(sys.argv[7])
penalty = sys.argv[8]
clippedGradient = sys.argv[9]
graph_option = True if sys.argv[10] == 'graph' else False
normalize = True if sys.argv[11] == 'normalize' else False

# read in data file
data = pd.read_excel(file)
columns = list(data.columns)

if normalize:
    for col in range(len(data)):
        if col in predictor_vars:
            data[columns[col-1]] = (data[columns[col-1]]-data[columns[col-1]].mean())/data[columns[col-1]].std()

# setup predictors
predictors = {}
for i in range(len(predictor_vars)): 
    predictors[i] = data[columns[predictor_vars[i]-1]][:training_range]
predictor = []
for i in predictors.keys():
    predictor = predictors[i]

# setup response
response = data[columns[response_var-1]][:training_range]

# setup test portion
test_predictors = {}
for i in range(len(predictor_vars)): 
    test_predictors[i] = [val for val in data[columns[predictor_vars[i]-1]][training_range:]]
test_response = [val for val in data[columns[response_var-1]][training_range:]]
# specific for univariate
test_predictor = []
if len(predictor_vars) == 1:
    for key in test_predictors.keys():
        test_predictor = test_predictors[key]

# run gradient descent algorithm
thetas = []
if len(predictor_vars) != 1:
    predictor_names = [columns[name-1] for name in predictor_vars]
    if regression_type == 'linear':
        cur_thetas,epoch_max,cur_mse,min_thetas,best_epoch,best_learning_rate,min_mse = multiLin(True if clippedGradient == 'clip' else False,predictors,response,training_range,learning_rates,epoch_max,True if penalty == 'penalty' else False,test_predictors,test_response,graph_option)
        displayMultivariate(True if len(learning_rates) > 1 else False,learning_rates,epoch_max,best_epoch,predictor_names,columns[response_var-1],training_range,data.shape[0]-training_range,best_learning_rate,cur_thetas,cur_mse,min_thetas,min_mse)
else: 
    if regression_type == 'linear': 
        cur_thetas,epoch_max,cur_mse,min_thetas,best_epoch,best_learning_rate,min_mse = uniLin(True if clippedGradient == 'clip' else False,predictor,response,training_range,learning_rates,epoch_max,True if penalty == 'penalty' else False,test_predictor,test_response,graph_option)
        displayUnivariate(True if len(learning_rates) > 1 else False,learning_rates,epoch_max,best_epoch,columns[predictor_vars[0]-1],columns[response_var-1],training_range,data.shape[0]-training_range,best_learning_rate,cur_thetas,cur_mse,min_thetas,min_mse)
    else:
        cur_thetas,epoch_max,cur_mse,min_thetas,best_epoch,best_learning_rate,min_mse = OLS(True if clippedGradient == 'clip' else False,predictor,response,training_range,learning_rates,epoch_max,True if penalty == 'penalty' else False,test_predictor,test_response,graph_option)
# calculate performance metrics



end_elapsed = t.time()
print(f"Elapsed Time: {end_elapsed - start_elapsed} sec")