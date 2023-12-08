import matplotlib.pyplot as plt
from parse import parse

if __name__== "__main__":
    # Load data
    X,Y = parse()

    # Names of input features
    columns = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
    
    # Create skatterplots comparing individual feature and burn area
    for i in range(12):
        column = columns[i]
        plt.scatter(X[:, i], Y)
        plt.title(column + ' compared to burned area')
        plt.xlabel(column)
        plt.ylabel('area')
        plt.show()
