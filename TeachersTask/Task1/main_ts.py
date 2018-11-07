# The file is to store the time series model with machine generated datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import datetime
from datetime import date
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


start_time = datetime.datetime.now()

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

print("Program starting. Please wait;)")
RoughDataset = ReadFile('wrfdata.5')

RoughDataSize = np.size(RoughDataset, 0)

TimeGen = pd.date_range(start="2015-01-01 8:00", end="2016-01-01 7:59", freq="5T")

SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

# count variable is to indicate how many number of 0 we've got so far
# count_line is to store the line number of zero result line
count = 0
count_line = list()

n = np.size(SWDIF, 0)
for i in range(n):
    if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
        count+=1
        count_line.append(i)

print("total number of data that need to be removed (night time): ",count)
count_line.sort()

for number in reversed(range(count)):
    line_number = count_line[number]
    SWDIR = np.delete(SWDIR, line_number, 0)
    SWDIF = np.delete(SWDIF, line_number, 0)
    GLW = np.delete(GLW, line_number, 0)
    TimeGen = TimeGen.drop(TimeGen[line_number])

CSR = SWDIF / (SWDIF + SWDIR)
dataSize = np.size(CSR, 0)
zeroVector = np.zeros(shape=(dataSize, ))

for i in range(dataSize):
    if CSR[i] < 0.1:
        zeroVector[i] = 1
    elif CSR[i] < 0.2:
        zeroVector[i] = 2
    elif CSR[i] < 0.3:
        zeroVector[i] = 3
    elif CSR[i] < 0.4:
        zeroVector[i] = 4
    elif CSR[i] < 0.5:
        zeroVector[i] = 5
    elif CSR[i] < 0.6:
        zeroVector[i] = 6
    elif CSR[i] < 0.7:
        zeroVector[i] = 7
    elif CSR[i] < 0.8:
        zeroVector[i] = 8
    elif CSR[i] < 0.9:
        zeroVector[i] = 9
    elif CSR[i] <= 1.0:
        zeroVector[i] = 10
    else:
        zeroVector[i] = 100

CorrepCSR = zeroVector

df = pd.DataFrame(TimeGen, columns=['date'])
df["CSR"] = CorrepCSR
print(df.head(15))
print("Shape of the data frame obtained: ", df.shape)

df_train, df_test = np.split(df, [36587], axis=0)

ts = df_train.set_index('date')
print("\nts's head: ", ts.head())

ts1 = ts.iloc[:, 0].values

# Below is a test to check the stationary of the data, link below
# https://machinelearningmastery.com/time-series-data-stationary-python/
result = adfuller(ts1)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

ts_log = np.log(ts)
ts_log_diff = ts_log - ts_log.shift()

model_AR = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model_AR.fit(disp=-1)

model_MA = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model_MA.fit(disp=-1)

model_ARIMA = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model_ARIMA.fit(disp=-1)

prediction_AR = model_AR.predict(start=df_test.iloc[0,0], end=df_test.iloc[15679, 0])
mse_AR = mean_squared_error(df_test.iloc[:,1], prediction_AR)
rmse_AR = math.sqrt(mse_AR)
print("RMSE of prediction_AR: ", rmse_AR)

prediction_MA = model_MA.predict(start=df_test.iloc[0,0], end=df_test.iloc[15679, 0])
mse_MA = mean_squared_error(df_test.iloc[:,1], prediction_MA)
rmse_MA = math.sqrt(mse_MA)
print("RMSE of prediction_MA: ", rmse_MA)

prediction_ARIMA = model_ARIMA.predict(start=df_test.iloc[0,0], end=df_test.iloc[15679, 0])
mse_ARIMA = mean_squared_error(df_test.iloc[:,1], prediction_ARIMA)
rmse_ARIMA = math.sqrt(mse_ARIMA)
print("RMSE of prediction_ARIMA: ", rmse_ARIMA)

end_time = datetime.datetime.now()
print("Time taken to run the program till complete the graph: ", (end_time-start_time).seconds, " seconds")