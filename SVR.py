import pprint
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from parse import parse

'''
Prints the results in a nice format for an SVR model for the 
specified kernel, grid of parameters, best parameters, and
test and training MSE
'''
def print_svroutput(kernel, params, best_params, test_mse, train_mse):
    pp = pprint.PrettyPrinter()

    print(f'-----SVR: {kernel} kernel-----')
    print("Parameters for grid search: " )
    pp.pprint(params)
    print()
    print("Best Parameters:", best_params)
    print("Test MSE:", test_mse)
    print("Train MSE", train_mse)
    print("\n")


'''
Does cross validation, tests, and outputs results for all the 
SVR kernels for the specified dataset and max epsilon value.
This is the coarse search/ first round of grid search
'''
def svr_coarse_search(dataset, epsilon_max):
    linear_params = {
        'C' : [1e-2, 1e-1, 1, 10, 100],
        'epsilon' : [epsilon_max/1000, epsilon_max/100, epsilon_max/10, epsilon_max]
    }

    svr("linear", linear_params, dataset)

    poly_params = {
        'C' : [1e-2, 1e-1, 1, 10, 100],
        'epsilon' : [epsilon_max/1000, epsilon_max/100, epsilon_max/10, epsilon_max],
        'degree' : [2,3,4]
    }

    svr("poly", poly_params, dataset)

    rbf_params = {
        'C' : [1e-2, 1e-1, 1, 10, 100],
        'epsilon' : [epsilon_max/1000, epsilon_max/100, epsilon_max/10, epsilon_max],
        'gamma':['auto','scale']
    }

    svr("rbf", rbf_params, dataset)

    sigmoid_params = {
        'C' : [1e-2, 1e-1, 1, 10, 100],
        'epsilon' : [epsilon_max/1000, epsilon_max/100, epsilon_max/10, epsilon_max],
        'gamma' : ['auto','scale']
    }

    svr("sigmoid", sigmoid_params, dataset)

'''
Does cross validation, tests, and outputs results for all the 
SVR kernels for the specified dataset and max epsilon value.
This is the granular search/ second round of grid search.
The values for the parameter grids are hard coded in this 
function. This program was first run just doing the coarse 
search, then the paramters for the granular search were updated
to be a tighter range centered around the result from the first
search.
'''
def svr_granular_search(dataset, epsilon_max):
    linear_params = {
        'C' : [0.3, 0.6, 1, 2.25, 4.5, 6.75],
        'epsilon' : [22*epsilon_max/100, 45*epsilon_max/100, 67*epsilon_max/100, epsilon_max/10, 3*epsilon_max/10, 6*epsilon_max/10]
    }

    svr("linear", linear_params, dataset)

    poly_params = {
        'C' : [0.03, 0.06, 1, .225, 0.45, 0.675],
        'epsilon' : [22*epsilon_max/100, 45*epsilon_max/100, 67*epsilon_max/100, epsilon_max/10, 3*epsilon_max/10, 6*epsilon_max/10],
        'degree' : [2]
    }

    svr("poly", poly_params, dataset)

    rbf_params =  {
        'C' : [3, 6, 10, 22.5, 45, 67.5],
        'epsilon' : [22*epsilon_max/100, 45*epsilon_max/100, 67*epsilon_max/100, epsilon_max/10, 3*epsilon_max/10, 6*epsilon_max/10],
        'gamma':['auto']
    }

    svr("rbf", rbf_params, dataset)

    sigmoid_params ={
        'C' : [0.3, 0.6, 1, 2.25, 4.5, 6.75],
        'epsilon' : [22*epsilon_max/100, 45*epsilon_max/100, 67*epsilon_max/100, epsilon_max/10, 3*epsilon_max/10, 6*epsilon_max/10],
        'gamma' : ['auto']
    }

    svr("sigmoid", sigmoid_params, dataset)

'''
Does grid search cross validation for a specified kernel, 
paramter grid and data set
'''
def svr(kernel, params, dataset):
    # Extract training and test data from dataset
    X_train = dataset[0]
    Y_train = dataset[1]

    X_test = dataset[2]
    Y_test = dataset[3]

    # Create model
    svr = SVR(kernel=kernel)
   
    # Create GridSearchCV object
    grid_search = GridSearchCV(svr, param_grid=params,  scoring='neg_mean_squared_error', n_jobs=-1, cv=5)

    # Fit to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    training_error = grid_search.best_score_
   
    # Use model to predict test data 
    Y_pred = best_estimator.predict(X_test)

    # Calculate test error
    mse = mean_squared_error(Y_test, Y_pred)
   
    print_svroutput(kernel, params, best_params, mse, training_error*-1)

   
if __name__== "__main__":
    # Load data
    X, Y = parse()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,
        random_state=42)
    
    # Calculate the maximum epsilon should be
    Y_max = Y_train.max()
    Y_min = Y_train.min()
    epsilon_max = (Y_max - Y_min)/2

    # Put all data arrays into one array - for easier function calls
    dataset = [X_train, Y_train, X_test, Y_test]

    # Do grid search cross validation
    svr_coarse_search(dataset, epsilon_max)

    print("============================================================")

    svr_granular_search(dataset, epsilon_max)
