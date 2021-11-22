# Gradient-Descent-Implementation

To test the program yourself follow these steps...

1. Prior to running package, ensure system has at least Python 3.7 downloaded and these packages...
- pandas
- numpy
- matplotlib
2. To run the program open the command-line interpreter for your respective machine (terminal for mac or cmd.exe for windows)
3. Then navigate to the location which holds the folder of files that were downloaded from the aforementioned link
4. program can be invoked with 'python main.py', however the program requires 11 additional command line arguments which include in order...
    - excel file name in quotes (string in quotes)
    - number of training instances (int)
    - linear (string)
    - what independent variables to use in quotes(list of comma separated ints)
    - what is the dependent variable or response (int)
    - learning rate in quotes (scientific notation float)
    - number of epochs or generations (int)
    - 'penalty' or 'no' (string)
    - 'clip' or 'no' (string)
    - 'graph' or 'no' (string)
    - 'normalize' or 'no' (string)
5. One example to input in command-line interpreter is 'python main.py 'Concrete_Data.xls' 900 linear '1,6,7' 9 '1.0E-8' 500 penalty no no no'
6. The example will run multivariate linear regression with 3 independent variables (cement component, coarse aggregate, fine aggregate) with a learning rate of 1.0E^-8, 500 generations, penalty, no gradient clipping, no graphing, and no normalizing
