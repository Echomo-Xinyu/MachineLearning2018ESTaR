# This file is to combine data from 63 stns and their correspond locations
# and apply them to a time series model.
import numpy as np
import datetime
import math
import warnings
import itertools
import pandas as pd
from pandas import datetime
import statsmodels.api as sm
from sklearn import svm
from sklearn import metrics
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from statsmodels.tsa.arima_model import ARMA
# from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


file_path = "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/tslist"
FinalDataset = np.zeros(shape=(2172480, 8), dtype=float)
location_nparray = np.zeros(shape=(63, 3), dtype=float)
def ReadList(file_pathway):
    f = open(file_pathway)
    line_number = 1
    i = 0
    for data in f.readlines():
        if line_number == 1 or line_number == 2 or line_number == 3:
            line_number+=1
        else:
            # remove the \n and space between data
            data = data.strip('\n')
            nums = data.split(' ')
            while '' in nums:
                nums.remove('')
            string = nums[1]
            index = string[3:]
            location_nparray[i,0] = index
            location_nparray[i,1], location_nparray[i,2] = nums[2], nums[3]
            i += 1
            if i == 63:
                break
    f.close()
ReadList(file_path)

startindex = "2015-01-01 8:00"
endindex = "2016-01-01 7:59"

TimeDF = pd.date_range(start=startindex, end=endindex, freq="15T")

# Read the data
def ReadFile(file_pathway):
    f = open(file_pathway)
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
k = 10
print("Current station index k is: ", k)
index = k + 1
file_path =  "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata." + str(index)
location = location_nparray[index]
startindex, endindex = 35040 * k, 35040 * (k+1)
FinalDataset[startindex:endindex, :3] = location

RoughDataset = np.asarray(ReadFile(file_path))
Time = RoughDataset[:, 0]
SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

CSR = SWDIF / (SWDIF + SWDIR)
dataSize = np.size(Time, 0)

CSRGroup = np.zeros(shape=(dataSize,)).transpose()

for i in range(dataSize):
    if np.isnan(CSR[i]):
        CSRGroup[i] = 0
    elif CSR[i] < 0.1:
        CSRGroup[i] = 1
    elif CSR[i] < 0.2:
        CSRGroup[i] = 2
    elif CSR[i] < 0.3:
        CSRGroup[i] = 3
    elif CSR[i] < 0.4:
        CSRGroup[i] = 4
    elif CSR[i] < 0.5:
        CSRGroup[i] = 5
    elif CSR[i] < 0.6:
        CSRGroup[i] = 6
    elif CSR[i] < 0.7:
        CSRGroup[i] = 7
    elif CSR[i] < 0.8:
        CSRGroup[i] = 8
    elif CSR[i] < 0.9:
        CSRGroup[i] = 9
    elif CSR[i] <= 1.0:
        CSRGroup[i] = 10
    else:
        print(i, ": ", CSR[i])
        CSRGroup[i] = 100
        print("Error! There is invalid value in CSR")
        print("The invalid value is in the line ", i)
    
for element in CSRGroup:
    if np.isnan(element):
        print("NaN present in CSR grouping")
        break


time_series = pd.DataFrame(CSRGroup, index=TimeDF)
# time_series = currentStnData.data

# # # Define the p, d and q parameters to take any value between 0 and 2
# # p = d = q = range(0, 2)

# # # Generate all different combinations of p, q and q triplets
# # pdq = list(itertools.product(p, d, q))

# # # Generate all different combinations of seasonal p, q and q triplets
# # seasonal_pdq = [(x[0], x[1], x[2], 2) for x in list(itertools.product(p, d, q))]

# # print('Examples of parameter combinations for Seasonal ARIMA...')
# # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# # warnings.filterwarnings("ignore") # specify to ignore warning messages

# # for param in pdq:
# #     for param_seasonal in seasonal_pdq:
# #         try:
# #             mod = sm.tsa.statespace.SARIMAX(time_series,
# #                                             order=param,
# #                                             seasonal_order=param_seasonal,
# #                                             enforce_stationarity=False,
# #                                             enforce_invertibility=False)

# #             results = mod.fit()

# #             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
# #         except:
# #             continue
# # output for the block of code above is some rediculous large number.. you won't wanna read them trust me
# # and the following (1, 0, 1, 2) is the optimal value with smallest AIC value

# mod = sm.tsa.statespace.SARIMAX(time_series,
#                                 order=(1, 0, 1),
#                                 seasonal_order=(1, 0, 1, 2),
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False)

# results = mod.fit()

# print(results.summary().tables[1])
# pred = results.get_prediction(start=pd.to_datetime("2015-01-01 8:00"), dynamic=False)
# y_forecasted = pred.predicted_mean
# y_truth = time_series["2015-01-01 8:00":]
# mse = ((pred - y_truth) ** 2).mean()
# print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

training_mod = sm.tsa.SARIMAX(time_series.loc[:'2015-12-31'], order=(1,0,1))
training_res = training_mod.fit()

mod = sm.tsa.SARIMAX(time_series, order=(1,0,1))
res = mod.filter(training_res.params)

# insample = res.predict()
# T = len(time_series.loc['2015-01-01':])
# forecast_error = np.abs(time_series.loc['2015-01-01':] - insample.loc['2015-01-01':])
# # insample.plot()
# # plt.show()
# print(np.sqrt(np.sum(forecast_error**2) / T))

y_train_pre = res.predict()
print("train: MAE: ", metrics.mean_absolute_error(time_series.loc["2015-01-01":], y_train_pre.loc["2015-01-01":]))
print("train: MSE: ",  metrics.mean_squared_error(time_series.loc["2015-01-01":], y_train_pre.loc["2015-01-01":]))
print("train: RMSE: ", np.sqrt(metrics.mean_squared_error(time_series.loc["2015-01-01":], y_train_pre.loc["2015-01-01":])))

# Output:
# train: MAE:  0.9541966341634895
# train: MSE:  4.322462402505193
# train: RMSE:  2.0790532466738783
