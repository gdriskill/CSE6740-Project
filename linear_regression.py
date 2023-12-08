from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from parse import parse

'''
Creates and tests a linear regression model for the specified
training and test data. Prints the training and test MSE.
'''
def linear_regression(X_train, X_test, Y_train, Y_test):
    # Create linear regression model
    regr = LinearRegression()

    # Fit model to training data
    regr.fit(X_train, Y_train)

    # Predict y values and calculate MSE
    Y_pred = regr.predict(X_test)
    error = mean_squared_error(Y_test, Y_pred)
    Y_pred_train = regr.predict(X_train)
    train_error = mean_squared_error(Y_train, Y_pred_train)
  
    # Print results
    print("----Linear regression---")
    print("Test MSE " + str(error))
    print("Traing MSE " + str(train_error))
    print("\n")

  
if __name__== "__main__":
    # Load data
    X, Y = parse()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,
        random_state=42)
    
    # Build and test linear regression model
    linear_regression(X_train, X_test, Y_train, Y_test)

    