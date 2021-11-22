from metrics import calcMSE
from graph import graphMSE, graphPrediction
import sys
import numpy as np
### allow user if they want a threshold
### otherwise no threshold and go to epoch max for each one


def getIterations(base_learning_rate,final_learning_rate):
    diff_var_one,diff_var_two = 0,0
    if(base_learning_rate>1.0E-5): diff_var_one = int(str(base_learning_rate).count('0'))
    else: diff_var_one = int(str(base_learning_rate)[str(base_learning_rate).index('-')+1:])
    if(final_learning_rate>1.0E-5): diff_var_two = int(str(final_learning_rate).count('0'))
    else: diff_var_two = int(str(final_learning_rate)[str(final_learning_rate).index('-')+1:])
    return abs(diff_var_one - diff_var_two)

def clipGradient(gradient,clip_min,clip_max):
    if(gradient>=clip_max): return clip_max
    elif (gradient<=clip_min): return clip_min
    else: return gradient

def getPenalty(lambda_1,lambda_2,theta,training_range):
    print('penalty')
    return ((lambda_1/training_range)*theta)+((2*lambda_2/training_range)*(theta**2))

def uniLin(clip,predictors,response,training_range,learning_rates,epoch_max,penalty,test_predictor,test_response,graph_option,min_thetas=[0,0]):
    theta_0_grad,theta_1_grad = 0,0
    best_epoch = 0
    best_learning_rate = 0
    min_mse = sys.maxsize
    Y = []
    cur_thetas = [0,0]
    for epoch in range(epoch_max):
        theta_0_grad,theta_1_grad = 0,0
        for instance in range(len(predictors)):
            error = response[instance]-(cur_thetas[1]*predictors[instance]+cur_thetas[0])
            theta_0_grad+=(-2*error)
            theta_1_grad+=((error)*(-2*(predictors[instance]))) 
        theta_0_grad/=(len(predictors))
        theta_1_grad/=(len(predictors))
        if clip: theta_0_grad,theta_1_grad = clipGradient(theta_0_grad,-0.5,0.5),clipGradient(theta_1_grad,-0.5,0.5)
        cur_thetas[0]-=(learning_rates[0]*theta_0_grad)
        cur_thetas[1]-=(learning_rates[0]*theta_1_grad)
        epochMSE = calcMSE(cur_thetas,'linear',False,test_predictor,test_response)
        if graph_option == True: Y.append(epochMSE)
    if graph_option == True:
        X = [i for i in range(1,epoch_max+1)]
        graphMSE(X,Y,learning_rates[0],'','')
        predicted = []
        for input in predictors: predicted.append(cur_thetas[0]+cur_thetas[1]*input)
        graphPrediction(predictors,predicted,response)
        print(f'MSE',Y.index(min(Y)),min(Y))
    print('MSE train',calcMSE(cur_thetas,'linear',False,predictors,response))
    return cur_thetas,epoch_max,epochMSE,min_thetas,best_epoch,learning_rates[0],min_mse

def multiLin(clip,predictors,response,training_range,learning_rates,epoch_max,penalty,test_predictors,test_response,graph_option):
    min_thetas,gradients = [0]*(len(predictors)+1), [0]*(len(predictors)+1)
    Y = []
    best_epoch = 0
    cur_thetas = [0]*(len(predictors)+1)
    min_mse = sys.maxsize
    lambda_1 = 1.0E6
    applied_penalty = (lambda_1/training_range) if penalty == True else 0
    for epoch in range(epoch_max):
        gradients = [0]*(len(predictors)+1)
        for instance in range(training_range):
            error=response[instance]-(cur_thetas[0]+sum([cur_thetas[cols]*predictors[cols][instance] for cols in predictors]))
            for gradient in range(len(gradients)): gradients[gradient]+=(-2*error) if gradient == 0 else (-2*predictors[gradient-1][instance]*error)
        for gradient in range(len(gradients)): gradients[gradient]/=training_range
        if clip:
            for gradient in range(len(gradients)): gradients[gradient] = clipGradient(gradients[gradient],-1,1)
        for theta in range(len(cur_thetas)): 
            if theta == 0: cur_thetas[theta] = (cur_thetas[theta])-(learning_rates[0]*(gradients[theta]))
            else: cur_thetas[theta] = (cur_thetas[theta])-(learning_rates[0]*(gradients[theta]+(applied_penalty*cur_thetas[theta])))
        cur_mse = calcMSE(cur_thetas,'linear',True,test_predictors,test_response)
        min_thetas = cur_thetas if cur_mse > min_mse else min_thetas
        min_mse = cur_mse if cur_mse > min_mse else min_mse
        best_epoch = (epoch+1) if cur_mse >min_mse else best_epoch
        if graph_option == True: Y.append(cur_mse)
    if graph_option == True:
        X = [i for i in range(1,epoch_max+1)]
        graphMSE(X,Y,learning_rates[0],'','')
    print('MSE train',calcMSE(cur_thetas,'linear',True,predictors,response))
    return cur_thetas,epoch_max,cur_mse,min_thetas,best_epoch,learning_rates[0],min_mse

# def multiPol(clip,predictors,response,training_range,learning_rates,epoch_max,threshold,test_predictors,test_response,graph_option):
#     min_thetas,gradients = [0]*(len(predictors)+1), [0]*(len(predictors)+1)
#     Y = []
#     best_epoch = 0
#     cur_thetas = [0]*(len(predictors)+1)
#     min_mse = sys.maxsize
#     cur_mse = 0
#     lambda_1 = 1.0E8
#     lambda_2 = 1.0E8
#     for epoch in range(epoch_max):
#         gradients = [0]*(len(predictors)+1)
#         for instance in range(training_range):
#             error=response[instance]-(cur_thetas[0]+sum([cur_thetas[cols]*(predictors[cols-1][instance]**cols) for cols in range(1,len(cur_thetas))]))
#             for gradient in range(len(gradients)): gradients[gradient]+=(-2*error) if gradient == 0 else (-2*(predictors[gradient][instance]**gradient)*error)
#         for gradient in range(len(gradients)): gradients[gradient]/=training_range
#         if clip:
#             for gradient in range(len(gradients)): gradients[gradient] = clipGradient(gradients[gradient],-1,1)
#         for theta in range(len(cur_thetas)): cur_thetas[theta] = cur_thetas[theta] - learning_rates[0]*gradients[theta]
#         cur_mse = calcMSE(cur_thetas,'polynomial',True,test_predictors,test_response)
#         min_thetas = cur_thetas if cur_mse > min_mse else min_thetas
#         min_mse = cur_mse if cur_mse > min_mse else min_mse
#         best_epoch = (epoch+1) if cur_mse >min_mse else best_epoch
#         if graph_option == True: Y.append(cur_mse)     
#     if graph_option == True:
#         X = [i for i in range(1,epoch_max+1)]
#         graphMSE(X,Y,learning_rates[0],'','')
#         print(f'MSE',Y.index(min(Y)),min(Y))
#     return cur_thetas,epoch_max,cur_mse,min_thetas,best_epoch,learning_rates[0],min_mse

def OLS(clip,predictors,response,training_range,learning_rates,epoch_max,threshold,test_predictors,test_response,graph_option): 
    print(predictors)
    x_squared_sum = sum([x**2 for x in predictors])
    x_sum = sum([x for x in predictors])
    y_sum = sum([y for y in response])
    xy_sum = sum([predictors[i]*response[i] for i in range(len(predictors))])
    theta_1 = ((len(predictors)*xy_sum)-(x_sum*y_sum))/((len(predictors)*x_squared_sum)-((x_sum)**2))
    theta_0 = (y_sum-theta_1*x_sum)/(len(predictors))
    thetas = [theta_0,theta_1]
    print('Train:',calcMSE(thetas,'linear',False,predictors,response))
    print('Test:',calcMSE(thetas,'linear',False,test_predictors,test_response))
    return 0,epoch_max,0,0,0,learning_rates[0],0
