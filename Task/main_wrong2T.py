'''
this mainT.py is to write a test version as I'm not sure what the algorithm should be like
in the current main.py is surpervised learning
yet we want a unsurpervised version
so now i'm writing a test version without any modification for the current surpervised version
'''

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import datetime

start_time = datetime.datetime.now()

bin_size = 0.1

# Read the data
def ReadFile(file_name):
    f = open(file_name)
    first_else = True
    line_number = 1
    for data in f.readlines():
        if line_number==1:
            line_number+=1
        else:
            # remove the \n and space between data
            data = data.strip('\n')
            nums = data.split(' ')
            if first_else:
                nums = [float(x) for x in nums]
                matrix = np.array(nums)
                first_else = False
            else:
                nums = [float(x) for x in nums]
                matrix = np.c_[matrix, nums]
    matrix = matrix.transpose()
    f.close()
    return matrix

print("Program running... Please wait")
Dataset = ReadFile('wrfdata.5')
# print(Dataset)
'''
now we have read all the data into the @Dataset variable for further preprocessing
'''

# Preprocessing the values

# Initialize the four variables we're gonna use
Time = Dataset[:, 0]
# print(Time)
SWDIR = Dataset[:, 1]
SWDIF = Dataset[:, 2]
GLW = Dataset[:, 3]

# count variable is to indicate how many number of 0 we've got so far
# count_line is to store the line number of zero result line
count = 0
count_line = list()

n = np.size(Time, 0)
for i in range(n):
    if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
        count+=1
        count_line.append(i)

print("total number of sets of useful dataset: ",count)

# @X_zero array is to store all the set of values during night time
X_zero = np.zeros(shape=(count, 4))

count_line.sort()
for number in reversed(range(count)):
    # if number > 0:
    #     number -= 1
    # try:
    line_number = count_line[number]
    X_zero[number][0] = Time[line_number]
    X_zero[number][1] = SWDIR[line_number]
    X_zero[number][2] = SWDIF[line_number]
    X_zero[number][3] = GLW[line_number]
    Time = np.delete(Time, line_number, 0)
    SWDIR = np.delete(SWDIR, line_number, 0)
    SWDIF = np.delete(SWDIF, line_number, 0)
    GLW = np.delete(GLW, line_number, 0)
    # except:
    #     print("number: ", number)
    #     print("line_number: ", line_number)
    #     print("X_zero.shape: ", X_zero.shape)

def checkRoughData():
    print("Time: ")
    print(Time)
    print("Time's shape: ", Time.shape)
    print("SWDIR: ")
    print(SWDIR)
    print("SWDIR's shape: ", SWDIR.shape)
    print("SWDIF: ")
    print(SWDIF)
    print("SWDIF's shape: ", SWDIF.shape)
    print("GLW: ")
    print(GLW)
    print("GLW's shape: ", GLW.shape)
    print("X_zero: ")
    print(X_zero)
    print("X_zero's shape:: ", X_zero.shape)
    for item in Time:
        if math.isnan(item):
            print("Nan exists in Time")
    for item in SWDIR:
        if math.isnan(item):
            print("Nan exists in SWDIR")
    for item in SWDIF:
        if math.isnan(item):
            print("Nan exists in SWDIF")
    for item in GLW:
        if math.isnan(item):
            print("Nan exists in GLW")
checkRoughData()

# To ignore the possible division by zero
# np.seterr(divide='ignore', invalid='ignore')
CSR = SWDIF/(SWDIR+SWDIF)
print("CSR: ")
print(CSR)
print("CSR's shape: ", CSR.shape)

def checkException():
    exception = list()

    for line_number in range(52267):
        item = CSR[line_number]
        if math.isnan(item):
            exception.append(line_number)
            print("line_number of nan: ", line_number)
    print("exception detected: ", exception)
    print("exception in Time: ", Time[exception[0]])
    print("exception in SWDIR: ", SWDIR[exception[0]])
    print("exception in SWDIF: ", SWDIF[exception[0]])
    print("exception in GLW: ", GLW[exception[0]])
# checkException()

# Combine time and CSR to one dataset
X = np.vstack((Time, CSR))
X = X.transpose()
print("X's shape: ", X.shape)

'''
so far we have completed the preprocessing of the data
and have two datasets @X and @X2
@X contains all the non-zero values for CSR which we'll be focusing on
@X_zero contains all the set with zero values for SWDIR & SWDIF
'''

'''
now we want to have a training set as well as a cross-validation
and by Coursera the best ratio is around 7 : 3
then we're gonna have 36588 training sets and 15680 cross-validation sets
'''

Xval = X[:15679, 0]
Yval = X[:15679, 1]
X = X[15679:, :]

def checkShape():
    print("Xval: ", Xval)
    print("Xval's shape: ", Xval.shape)
    print("Yval: ", Yval)
    print("Yval's shape: ", Yval.shape)
    print("X: ", X)
    print("X's shape: ", X.shape)
checkShape()
Xsize = np.size(X[0])
# y = X[:, 1]
# X = X[:, 0]

# Fit the dataset to the Gaussian model
# mu = np.mean(X, axis=0)
sigma2 = np.std(X, ddof=0, axis=0)

# print("mu: ", mu)
# # print("mu's shape: ", mu.shape)
# print("sigma2: ", sigma2)
# # print("sigma2's shape: ", sigma2.shape)
# print('\n')


# A test for making sure all sigma2 values are positive
def checkPositive(Array):
    num = 0
    for item in Array:
        if item < 0:
            num +=1
    print("The number of negative in the array: ", num)
    print('\n')

def checkNaN(Array):
    num = 0
    for item in Array:
        if math.isnan(item):
            # print("What the hell")
            num +=1
    print("The number of NaN in the array: ", num)
    print('\n')

# sigma = np.sqrt(sigma2)

def checkNaNException():
    step = np.sqrt(sigma2) 
    if np.isnan(step).any(): 
        # do something to report the error 
        badItems = np.where(np.isnan(step)) 
        print("bad inputs at :"+str(badItems))
        print("bad input values: "+str(sigma2[np.isnan(step)]))
        raise Exception("unexpected nans in sqrt step") 



# The function below is for the X with more than one values
# and don't work for our case as we only have time as parameter
def multivariateGaussian(X_input):
    epsilon_rough = 1
    # coefficient = 1 / (np.power(math.sqrt(2*math.pi), n_ts) * sigma)
    mu = np.mean(X, axis=0)
    sigma2 = np.std(X, ddof=0, axis=0)
    sigma = np.sqrt(sigma2)
    try:
        size_X, useless_variable = X_input.shape
        print("useless variable: (you are expecting to see a 2) ", useless_variable)
    except:
        size_X, = X_input.shape
    
    for i in range(size_X):
        X_input[i] = X_input[i] - mu
        coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)
        secondPart = np.exp(- np.power(X_input[i], 2) / (2 * sigma2))
        epsilon_rough = epsilon_rough * coefficient * secondPart
    # for a in range(10):
    #     print(X[a,:])
    return epsilon_rough
    # return 1 / ((2 * math.pi) ** ( - n_ts / 2) * sigma) * (np.exp(- np.power(X-mu, 2) / (2 *sigma2))).transpose()

pval = multivariateGaussian(X)

print("epsilon: ", pval)

def selectThreshold(yval, pval):
    bestEpsilon = 0.0
    bestF1 = 0.0
    stepsize = (np.max(yval) - np.min(yval)) / 1000
    print(stepsize)
    for epsilon in range(0, 1, 1000):
        yval1 = yval == 1
        num_y1 = yval[yval1].size
        yval0 = yval == 0
        num_y0 = yval[yval0].size
        pvals = pval < epsilon
        num_ps = pval[pvals].size
        pvall = pval >= epsilon
        num_pl = pval[pvall].size

        tp = num_y1 + num_ps
        fp = num_y0 + num_ps
        fn = num_y1 + num_pl
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = (2 * prec * rec) / (prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

def visualize(X):
    x_para = X[:,0]
    y_para = X[:,1]
    plt.plot(x_para, y_para, 'b,', label='data')
    x_fit = np.linspace(np.min(x_para), np.max(y_para), 1000)
    # I'm really confused about how to draw the graph
    plt.plot(x_fit, multivariateGaussian(X), 'r-', label='fit')
    plt.legend()
    plt.title('Fig 3')
    plt.xlabel('Time')
    plt.ylabel('CSR')
    plt.show()

# def plotGraph(x_para, y_para):
#     popt, pcov = curve_fit(gaussian, x_para, y_para, p0=[np.max(y_para), np.median(x_para), np.std(x_para), np.min(y_para)])
#     # plot original data
#     plt.plot(x_para, y_para, 'b,', label='data')
#     # plot fit function
#     x_fit = np.linspace(np.min(x_para), np.max(x_para), 1000)
#     plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label='fit')
#     plt.legend()
#     plt.title('Fig 1')
#     plt.xlabel('Time')
#     plt.ylabel('CSR')

#     end_time = datetime.datetime.now()
#     print("Time taken to run the program till complete the graph: ", (end_time-start_time).seconds, " seconds")

#     plt.show()


# pval = gaussian(Xval, Xsize, mu, np.sqrt(sigma2))
[epsilon, F1] = selectThreshold(Yval, pval)
print("Best epsilon found using cross-validation: ", epsilon)
print("Best F1 on cross-validation set: ", F1)

visualize(X)