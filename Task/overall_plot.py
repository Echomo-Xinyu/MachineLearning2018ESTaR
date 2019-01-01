import math
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier

start_time = datetime.datetime.now()

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

print("Program starting. Please wait;)")

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=15)
result = []
result2 = []
result3 = []
while dt < end:
    result.append(float(dt.strftime('%m%d%H%M')))
    result2.append(dt.strftime("%H:%M"))
    result3.append(dt.strftime("%m"))
    dt += step
Time = np.asarray(result)
DateTime = np.asarray(result2)
month_index = np.asarray(result3)

# # knn
# FinalDataset = np.zeros(shape=(2172480, 8), dtype=float)
# location_nparray = np.zeros(shape=(63, 3), dtype=float)

# # mean CSR value each month
# dayEachMonth = {1:"31", 2:"28", 3:"31", 4:"30", 5:"31", 6:"30", 7:"31", 8:"31", 9:"30", 10:"31", 11:"30", 12:"31"}
# def TotalDayInMonth(monthIndex):
#     day_number = 0
#     if monthIndex == 0:
#         return day_number
#     while monthIndex != 1:
#         day_number += int(dayEachMonth[monthIndex])
#         monthIndex -= 1
#     day_number += int(dayEachMonth[1])
#     return day_number

# for k in range(62):
#     index = k + 1
#     file_path =  "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata." + str(index)
#     location = location_nparray[index]
#     startindex, endindex = 35040 * k, 35040 * (k+1)
#     FinalDataset[startindex:endindex, :3] = location

#     RoughDataset = np.asarray(ReadFile(file_path))
#     SWDIR = RoughDataset[:, 1]
#     SWDIF = RoughDataset[:, 2]
#     GLW = RoughDataset[:, 3]

#     CSR = SWDIF / (SWDIF + SWDIR)
#     dataSize = np.size(Time, 0)

#     ProcessedDataset = np.vstack((Time, month_index, CSR))
#     ProcessedDataset = ProcessedDataset.transpose()

#     zeroVector = np.zeros(shape=(1, dataSize))
#     # 3 for day average value, 4 for month average value, 5 for CSR grouping
#     ProcessedDataset = np.insert(ProcessedDataset, 3, values=zeroVector, axis=1)
#     ProcessedDataset = np.insert(ProcessedDataset, 4, values=zeroVector, axis=1)
#     ProcessedDataset = np.insert(ProcessedDataset, 5, values=zeroVector, axis=1)
#     # below is to convert ratio CSR to correspond group 0-10
#     for element in ProcessedDataset:
#         if element[2] < 0.1:
#             element[5] = 1
#         elif element[2] < 0.2:
#             element[5] = 2
#         elif element[2] < 0.3:
#             element[5] = 3
#         elif element[2] < 0.4:
#             element[5] = 4
#         elif element[2] < 0.5:
#             element[5] = 5
#         elif element[2] < 0.6:
#             element[5] = 6
#         elif element[2] < 0.7:
#             element[5] = 7
#         elif element[2] < 0.8:
#             element[5] = 8
#         elif element[2] < 0.9:
#             element[5] = 9
#         elif element[2] <= 1.0:
#             element[5] = 10
#         else:
#             element[5] = 0
#     # mean CSR value each day
#     for j in range(365):
#         indexa, indexb = j * 365, (j+1) * 365
#         ProcessedDataset[indexa:indexb, 3] = np.mean(ProcessedDataset[indexa:indexb, 5])
#     # mean CSR value each month
#     for i in range(12):
#         MonthIndex = i + 1
#         DayNumber1, DayNumber2 = 0, 0
#         DayNumber1 = TotalDayInMonth(i)
#         DayNumber2 = TotalDayInMonth(MonthIndex)
#         if i == 11:
#             ProcessedDataset[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset[DayNumber1:, 5])
#             break
#         ProcessedDataset[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset[DayNumber1:DayNumber2, 5])
#     # ProcessedDataset should contain 5 columns:
#     # time long float, month index, day mean CSR, month mean CSR, actual CSR group
#     ProcessedDataset = np.delete(ProcessedDataset, 2, 1)
#     # print("Shape of ProcessedDataset: ", ProcessedDataset.shape)
#     # print("Shape of fit in final dataset: ", FinalDataset[startindex:endindex, 3:8].shape)
#     FinalDataset[startindex:endindex, 3:8] = ProcessedDataset
#     if k == 54:
#         PredictionDataset = FinalDataset[startindex:endindex, :].copy()
#     print("Current range index k is: ", k)

# print("Great to know you have completed the read in of dataset!!!")
# print("Now is the more time consuming part..")
# FinalDataset_copy = FinalDataset.copy()
# np.random.shuffle(FinalDataset_copy)
# print("Shape of final dataset: ", FinalDataset.shape)
# X_sample, y_sample = FinalDataset_copy[:, :7], FinalDataset_copy[:, 7]
# x_true, y_true = FinalDataset[:, :7], FinalDataset[:, 7]
# x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.1, random_state=42)

# print("Now begin to fit in the data.")
# start_time = datetime.datetime.now()

# reg = KNeighborsRegressor(n_neighbors=400)
# reg.fit(x_train, y_train)

# print("Data fit in successfully!")

# end_time = datetime.datetime.now()
# print("Time taken to run the program till fit in of all the data: ", (end_time-start_time).seconds, " seconds")

# y_true_pre_knn = reg.predict(x_true)
# x_prediction, y_prediction = PredictionDataset[:, :7], PredictionDataset[:, 7]
# y_true_pre_knn = reg.predict(x_prediction)
# plt.plot(DateTime[:64], y_true_pre_knn[:64], 'k--', linewidth=1, label="KNN model")



# svm and rf
RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.55')

SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

CSR = SWDIF / (SWDIF + SWDIR)
print("CSR:\n", CSR)

dataSize = np.size(Time, 0)
print("total number of useful datas is: ", dataSize)

ProcessedDataset = np.vstack((Time, CSR))
ProcessedDataset = ProcessedDataset.transpose()
print("Processed dataset: \n", ProcessedDataset)
print("Processed dataset's shape: ", ProcessedDataset.shape, "\n\n")

zeroVector = np.zeros(shape=(1, dataSize))
ProcessedDataset = np.insert(ProcessedDataset, 2, values=zeroVector, axis=1)

for element in ProcessedDataset:
    if element[1] < 0.1:
        element[2] = 1
    elif element[1] < 0.2:
        element[2] = 2
    elif element[1] < 0.3:
        element[2] = 3
    elif element[1] < 0.4:
        element[2] = 4
    elif element[1] < 0.5:
        element[2] = 5
    elif element[1] < 0.6:
        element[2] = 6
    elif element[1] < 0.7:
        element[2] = 7
    elif element[1] < 0.8:
        element[2] = 8
    elif element[1] < 0.9:
        element[2] = 9
    elif element[1] <= 1.0:
        element[2] = 10
    else:
        element[2] = 0
ProcessedDataset = np.delete(ProcessedDataset, 1, 1)

ProcessedDataset_copy = ProcessedDataset.copy()
np.random.shuffle(ProcessedDataset_copy)
print("Shape of the processed dataset: ", ProcessedDataset.shape)

X_sample, y_sample = ProcessedDataset_copy[:, 0].reshape(-1,1), ProcessedDataset_copy[:,1]
x_true, y_true = ProcessedDataset[:64, 0].reshape(-1, 1), ProcessedDataset[:64, 1]
x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.1, random_state=42)

#@C is 1/alpha and can be used to regulate the function
clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train)
y_true_pre_svm = clf.predict(x_true)
plt.figure(figsize=(50, 13.596))
plt.plot(DateTime[:64], y_true_pre_svm[:], 'b--', linewidth=1, label="SVC model")
plt.plot(DateTime[:64], y_true[:], 'r+', label="Actual observations")

# RF Block
X_sample, y_sample = ProcessedDataset_copy[:, 0].reshape(-1,1), ProcessedDataset_copy[:,1]
x_true, y_true = ProcessedDataset[:, 0].reshape(-1, 1), ProcessedDataset[:, 1]
x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.25, random_state=13)
rf = RandomForestClassifier(n_estimators=1000, random_state=23)
rf.fit(x_train, y_train)
y_true_pre_rf = rf.predict(x_true)
plt.plot(DateTime[:64], y_true_pre_svm[:], 'm--', linewidth=1, label="RF model")

# SARIMAX block
startindex = "2015-01-01 8:00"
endindex = "2016-01-01 7:59"
TimeDF = pd.date_range(start=startindex, end=endindex, freq="15T")

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

time_series = pd.DataFrame(CSRGroup, index=TimeDF)
training_mod = sm.tsa.SARIMAX(time_series.loc[:'2015-12-31'], order=(1,0,1))
training_res = training_mod.fit()
mod = sm.tsa.SARIMAX(time_series, order=(1,0,1))
res = mod.filter(training_res.params)

y_train_pre = res.predict()
plt.plot(DateTime[:64], y_train_pre[:64], 'y--', linewidth=1, label="SARIMAX model")

plt.title("CSR prediction by SVC, RF, KNN, and SARIMAX models on 1st January 2018 at Tuas South")
plt.xlabel("Time")
plt.ylabel("CSR value")
plt.legend()
plt.savefig("DAY1_overall_plots.svg", format="svg")
plt.close()