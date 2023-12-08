# code modified from class github https://github.com/ni-sha-c/CSE-6740-Fall23/blob/main/code/fc.py
# and also the tutorial: https://machinelearningmastery.com/how-to-grid-search-hyperparameters-for-pytorch-models/
# implement fully connected neural network with pytorch
# Mostly written by chatGPT
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pprint
from parse import parse


# Define a simple feedforward neural network
class FeedForwardNet(nn.Module):
    def __init__(self, input_size=12, hidden_size=16, output_size=1):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (if not already flattened)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

'''
Creates a feed forward network and does grid search cross
validation for the specifed paramter grid and dataset
'''
def NN_search(param_grid, dataset):
    # Extract training and test data from dataset
    X_train = dataset[0]
    y_train = dataset[1]

    X_test = dataset[2]
    y_test = dataset[3]

    # Create the Skorch regressor
    model = NeuralNetRegressor(
        FeedForwardNet,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        verbose=False
    )

    # Create GridSearchCV object
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
   
    # Fit to the data
    grid_result = grid.fit(X_train, y_train)
    
    # Get the best parameters and best estimator
    train_mse = grid_result.best_score_*-1
    best_params = grid_result.best_params_
    best_estimator = grid_result.best_estimator_

    # Use model to predict test data 
    y_pred = best_estimator.predict(X_test)

    # Calculate test error
    test_mse = mean_squared_error(y_test, y_pred)

    print_nnoutput(param_grid, best_params, test_mse, train_mse)


'''
Prints the results in a nice format for a NNfor the 
specified kernel, grid of parameters, best parameters, and
test and training MSE
'''
def print_nnoutput(param_grid, best_params, test_mse, train_mse):
   
    pp = pprint.PrettyPrinter()
    print("----ANN----")
    print("Paramters for grid search:")
    pp.pprint(param_grid)
    print("Best Parameters: ", best_params)
    print("Test MSE:", test_mse)
    print("Train MSE", train_mse)
    print()


if __name__ == "__main__":
    # Load data
    X, Y = parse()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, shuffle=False)

    # Convert from numpy arrays to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    dataset = [X_train, y_train, X_test, y_test]

    # Define parameter grid for coarse search
    param_gridcoarse = {
        'batch_size': [10, 20, 30, 40],
        'max_epochs': [50, 100, 150],
        'optimizer__lr': [0.001, 0.01, 0.1, 0.2],
        'module__hidden_size': [6,12,25]
    }

    # Create and test NN wtih coarse search
    NN_search(param_gridcoarse)
    
    param_gridgranular = {
        'batch_size': [35, 40, 45, 50],
        'max_epochs': [40, 45, 50],
        'optimizer__lr': [0.0001, 0.005, 0.001],
        'module__hidden_size': [4, 5, 6]
    }

    # Create and test NN with granular search
    NN_search(param_gridgranular)
