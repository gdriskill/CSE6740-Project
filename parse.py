import numpy as np
from sklearn import preprocessing as pre

def parse():
    arr = np.loadtxt("./data/forestfires.csv", delimiter=",", converters={2 : month_to_num, 3 : day_to_num}, skiprows=1, encoding=None)
    X = arr[:, :-1]
    Y = arr[:, -1]
    Y = np.log(Y + 1) # transformation suggested by paper
    
    Y = Y.reshape(-1,1)
    
    normalizer = pre.MinMaxScaler()

    X = normalizer.fit_transform(X)
    Y = normalizer.fit_transform(Y)

    Y = np.squeeze(Y)

    return X, Y

def day_to_num(day):
    if day == "mon":
        retval = 1
    elif day == "tue":
        retval = 2
    elif day == "wed":
        retval = 3
    elif day == "thu":
        retval = 4
    elif day == "fri":
        retval = 5
    elif day == "sat":
        retval = 6
    elif day == "sun":
        retval = 7
    
    return retval

def month_to_num(month):
    if month == "jan":
        retval = 1
    elif month == "feb":
        retval = 2
    elif month == "mar":
        retval = 3
    elif month == "apr":
        retval = 4    
    elif month == "may":
        retval = 5
    elif month == "jun":
        retval = 6
    elif month == "jul":
        retval = 7
    elif month == "aug":
        retval = 8
    elif month == "sep":
        retval = 9
    elif month == "oct":
        retval = 10
    elif month == "nov":
        retval = 11
    elif month == "dec":
        retval = 12

    return retval