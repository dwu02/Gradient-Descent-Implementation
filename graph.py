import matplotlib.pyplot as plt

def graphMSE(X,Y,learning_rate,predictor_name,response_name):
    plt.title(f'MSE for alpha of {learning_rate}')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.plot(X,Y)
    plt.show()

def graphPrediction(X,Y,response):
    #Y is the line
    plt.title(f'Trained Model on Training Data')
    plt.scatter(X,response)
    plt.plot(X,Y)
    plt.show()
