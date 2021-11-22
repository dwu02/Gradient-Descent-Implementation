def calcMSE(thetas,regression_type,run_type,predictors,response):
    predicted = []
    if run_type == True: # multivariate
        total_squared_error = 0
        cur_predictions = []
        if regression_type == 'linear':
            for i in range(len(response)):
                cur_prediction = thetas[0]+sum([thetas[cols]*predictors[cols-1][i] for cols in range(1,len(thetas))])
                total_squared_error+=((cur_prediction-response[i])**2)
                cur_predictions.append(cur_prediction)
        else:
            exponents = [exponent for exponent in range(len(thetas)+1)]
            for i in range(len(response)):
                cur_prediction=0
                cur_prediction+=thetas[0]
                for cols in range(1,len(thetas)):
                    cur_prediction+=(thetas[cols]*(predictors[cols][i]**cols))
                total_squared_error+=((cur_prediction-response[i])**2)
                cur_predictions.append(cur_prediction)
        return total_squared_error/(len(response))
    else: # univariate
        for input in predictors: predicted.append(thetas[0]+thetas[1]*input)
        cur_sum = 0
        response = [val for val in response]
        for i in range(len(predicted)): 
            cur_sum+=((predicted[i]-response[i])**2)
        return cur_sum/len(predictors)
