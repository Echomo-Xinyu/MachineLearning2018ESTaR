'''
This .py is to collect all the days with low CSR ( < 0.1 )
c for collect
'''

'''
This script is under the instructions from Dr Harish. The instructions are below:
A single model will not work for the entire dataset.
If we had clear skies daily, the Gaussian model will be a good fit to the whole data collection.
You need to apply the model for every 24 hours. (leave out the first 16 hours).
If the correlation is greater than ~ 0.75, you can assume that the fit is good.
However, for days in which there are cloudy skies (common in Singapore),
you need to detect the sudden drop in the radiation and factor it approximately into the model.
This will minimize the error of your predictions.
The skeleton for model construction reads as follows:

Direct = f (time, cloudiness)
Diffuse = g(time,cloudiness)

where cloudiness = Diffuse / (Direct + Diffuse) from the data given to you.
When cloudiness is less than 0.2, you can use a Gaussian fit for both direct and diffuse.
Beyond this cloudiness, you need to think of a way for constructing the model.
You need to start exploring the different model construction options 
and start thinking of ways to represent the data.
For example, you can look into how Perez Model reconstructs solar radiation.
Also the common tools for solar radiation prediction are the hidden Markov model and support vector regression.
You need to think of a way of using the data with these models 
and let us know if you need additional inputs.
'''

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math

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
RoughDataset = ReadFile('wrfdata.5')

Time = RoughDataset[:,0]
SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

# dataSize = np.size(Time, 0)
# print("total number of useful datas is: ", dataSize)

# count variable is to indicate how many number of 0 we've got so far
# count_line is to store the line number of zero result line
count = 0
count_line = list()

dataSize = np.size(Time, 0)
for i in range(dataSize):
    if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
        count+=1
        count_line.append(i)

print("total number of useless datasets: ",count)
print("total number of useful sets of data is: ", dataSize-count)

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
    print("Time:\n",Time)
    print("Time's shape: ", Time.shape)
    print("SWDIR:\n",SWDIR)
    print("SWDIR's shape: ", SWDIR.shape)
    print("SWDIF:\n",SWDIF)
    print("SWDIF's shape: ", SWDIF.shape)
    print("GLW:\n",GLW)
    print("GLW's shape: ", GLW.shape)
    print("X_zero:\n",X_zero)
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
# checkRoughData()

CSR = SWDIF / (SWDIF + SWDIR)
# print("CSR:\n", CSR)
print("CSR's shape: ", CSR.shape)

# I decide to have the data for the first month
startTime = 16
endTime = startTime + 24 * 30

Time_Month_1 = Time[startTime:endTime]
CSR_Month_1 = CSR[startTime:endTime]


X_Month1 = np.vstack((Time_Month_1, CSR_Month_1))
X_Month1 = X_Month1.transpose()
print("X's shape for month 1: ", X_Month1.shape)

def scatterplot(x_data, y_data, x_label="Time", y_label="CSR", title="Dataset for the first month", color = "r", yscale_log=False):
    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    plt.scatter(x_data, y_data, s = 10, color = color, alpha = 0.7)

    # Label the axes and provide a title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# scatterplot(Time_Month_1, CSR_Month_1)

# wait = input("Press ENTER to continue")
# print("Now we are going to collect the most clear days' data from the whole month")
usefulDataSize = np.size(Time, 0)

clear_count = 0
clear_days_CSR = list()
clear_days_time = list()
clear_days_SWDIF = list()
clear_days_SWDIR = list()
for i in range(usefulDataSize):
    if CSR[i] <= 0.1:
        clear_count += 1
        clear_days_CSR.append(CSR[i])
        clear_days_time.append(Time[i])
        clear_days_SWDIF.append(SWDIF[i])
        clear_days_SWDIR.append(SWDIR[i])


clear_days_Data = np.vstack((clear_days_time, clear_days_CSR, clear_days_SWDIF, clear_days_SWDIR))
clear_days_Data = clear_days_Data.transpose()
print("Dataset's shape for clear days: ", clear_days_Data.shape)

def singleGaussian(x, a, mu, sig, offset=0):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + offset

def fitDataIntoModel(x_para, y_para, given_model=singleGaussian):
    plt.plot(x_para, y_para, 'b,', label='data')
    popt, pcov = curve_fit(given_model, x_para, y_para)
    x_fit = np.linspace(np.min(x_para), np.max(x_para), 1000)
    optimalValue = given_model(x_fit, *popt)
    # print("optimal value: ", optimalValue)
    plt.plot(x_fit, optimalValue, 'r-', label='fit')
    plt.legend()
    plt.xlabel('Time')
    plt.show()
    
# fitDataIntoModel(clear_days_time, clear_days_SWDIF)

# The function below is for the X with more than one values
# and don't work for our case as we only have time as parameter
def multivariateGaussian(X_input):
    epsilon_rough = 1
    # coefficient = 1 / (np.power(math.sqrt(2*math.pi), n_ts) * sigma)
    mu = np.mean(X_input, axis=0)
    sigma2 = np.std(X_input, ddof=0, axis=0)
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
    return epsilon_rough

CD_Time_SWDIF = np.vstack((clear_days_time, clear_days_SWDIF))
CD_Time_SWDIF = CD_Time_SWDIF.transpose()
print("CD_Time_SWDIF's shape: ", CD_Time_SWDIF.shape)

p = multivariateGaussian(CD_Time_SWDIF)
print("Epsilon: ", p)

def multivariatePlot(x_para, y_para, given_model=multivariateGaussian):
    z_para = np.zeros([np.size(x_para)])
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x_para, y_para, z_para, 'b,')
    # popt, pcov = curve_fit(given_model, )



