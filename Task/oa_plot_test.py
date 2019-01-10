# The file is to complete the overall plot
import math
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier

print("The program running now is a model based on KNN regressor.")
file_path = "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/tslist"
FinalDataset_knn = np.zeros(shape=(2172480, 8), dtype=float)
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

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=15)
result1 = []
result2 = []
result3 = []
# result4 = []
while dt < end:
    result1.append(float(dt.strftime('%Y%m%d%H%M%S')))
    result2.append(float(dt.strftime('%m')))
    result3.append(float(dt.strftime("%m%d%H%M")))
    # result4.append(dt.strftime("%H:%M"))
    dt += step
Time_long = np.asarray(result1)
month_index = np.asarray(result2)
Time = np.asarray(result3)
# DateTime = np.asarray(result4)

# DateTime = np.arange('2018-01-01T08:00', '2019-01-01T07:59', dtype='datetime64[D]')

startindex = "2015-01-01 8:00"
endindex = "2016-01-01 7:59"
DateTime = pd.date_range(start=startindex, end=endindex, freq="15T")

# mean CSR value each month
dayEachMonth = {1:"31", 2:"28", 3:"31", 4:"30", 5:"31", 6:"30", 7:"31", 8:"31", 9:"30", 10:"31", 11:"30", 12:"31"}
def TotalDayInMonth(monthIndex):
    day_number = 0
    if monthIndex == 0:
        return day_number
    while monthIndex != 1:
        day_number += int(dayEachMonth[monthIndex])
        monthIndex -= 1
    
    day_number += int(dayEachMonth[1])
    return day_number

for k in range(62):
    index = k + 1
    file_path =  "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata." + str(index)
    location = location_nparray[index]
    startindex, endindex = 35040 * k, 35040 * (k+1)
    FinalDataset_knn[startindex:endindex, :3] = location

    RoughDataset_knn = np.asarray(ReadFile(file_path))
    SWDIR = RoughDataset_knn[:, 1]
    SWDIF = RoughDataset_knn[:, 2]
    GLW = RoughDataset_knn[:, 3]

    CSR = SWDIF / (SWDIF + SWDIR)
    dataSize = np.size(Time_long, 0)

    ProcessedDataset_knn = np.vstack((Time_long, month_index, CSR))
    ProcessedDataset_knn = ProcessedDataset_knn.transpose()

    zeroVector = np.zeros(shape=(1, dataSize))
    # 3 for day average value, 4 for month average value, 5 for CSR grouping
    ProcessedDataset_knn = np.insert(ProcessedDataset_knn, 3, values=zeroVector, axis=1)
    ProcessedDataset_knn = np.insert(ProcessedDataset_knn, 4, values=zeroVector, axis=1)
    ProcessedDataset_knn = np.insert(ProcessedDataset_knn, 5, values=zeroVector, axis=1)
    # below is to convert ratio CSR to correspond group 0-10
    for element in ProcessedDataset_knn:
        if np.isnan(element[2]):
            element[5] = 0
        elif element[2] < 0.1:
            element[5] = 1
        elif element[2] < 0.2:
            element[5] = 2
        elif element[2] < 0.3:
            element[5] = 3
        elif element[2] < 0.4:
            element[5] = 4
        elif element[2] < 0.5:
            element[5] = 5
        elif element[2] < 0.6:
            element[5] = 6
        elif element[2] < 0.7:
            element[5] = 7
        elif element[2] < 0.8:
            element[5] = 8
        elif element[2] < 0.9:
            element[5] = 9
        else:
            element[5] = 10
    # mean CSR value each day
    for j in range(365):
        indexa, indexb = j * 96, (j+1) * 96
        ProcessedDataset_knn[indexa:indexb, 3] = np.mean(ProcessedDataset_knn[indexa:indexb, 5])
    # mean CSR value each month
    for i in range(12):
        MonthIndex = i + 1
        DayNumber1, DayNumber2 = 0, 0
        DayNumber1 = TotalDayInMonth(i)
        DayNumber2 = TotalDayInMonth(MonthIndex)
        if i == 11:
            ProcessedDataset_knn[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset_knn[DayNumber1:, 5])
            break
        ProcessedDataset_knn[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset_knn[DayNumber1:DayNumber2, 5])
    # ProcessedDataset should contain 5 columns:
    # time long float, month index, day mean CSR, month mean CSR, actual CSR group
    ProcessedDataset_knn = np.delete(ProcessedDataset_knn, 2, 1)
    # print("Shape of ProcessedDataset: ", ProcessedDataset.shape)
    # print("Shape of fit in final dataset: ", FinalDataset[startindex:endindex, 3:8].shape)
    FinalDataset_knn[startindex:endindex, 3:8] = ProcessedDataset_knn
    if k == 54:
        PredictionDataset_knn = FinalDataset_knn[startindex:endindex, :].copy()
    print("Current range index k is: ", k)

print("Great to know you have completed the read in of dataset!!!")
print("Now is the more time consuming part..")
FinalDataset_knn_copy = FinalDataset_knn.copy()
np.random.shuffle(FinalDataset_knn_copy)
print("Shape of final Dataset_knn: ", FinalDataset_knn.shape)
x_sample_knn, y_sample_knn = FinalDataset_knn_copy[:, :7], FinalDataset_knn_copy[:, 7]
x_train_knn, x_test, y_train_knn, y_test = train_test_split(x_sample_knn, y_sample_knn, test_size=0.1, random_state=42)

print("Now begin to fit in the data.")
start_time = datetime.datetime.now()

reg = KNeighborsRegressor(n_neighbors=400)
reg.fit(x_train_knn, y_train_knn)

print("Data fit in successfully!")

end_time = datetime.datetime.now()
print("Time taken to run the program till fit in of all the data: ", (end_time-start_time).seconds, " seconds")

x_prediction, y_prediction = PredictionDataset_knn[:, :7], PredictionDataset_knn[:, 7]
y_prediction_pre = reg.predict(x_prediction)
plt.figure(figsize=(20, 13.596))

# Dt_formatted = [datetime.datetime.strptime(d, "%H:%M").date() for d in DateTime]
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H;%M'))

plt.plot(DateTime[:64], y_prediction[:64], 'r+', label="Actual observation")
plt.plot(DateTime[:64], y_prediction_pre[:64], 'b--', linewidth=1, label="KNN model")

# svm
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
print("Processed Dataset: \n", ProcessedDataset)
print("Processed Dataset's shape: ", ProcessedDataset.shape, "\n\n")

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
print("Shape of the processed Dataset: ", ProcessedDataset.shape)

X_sample, y_sample = ProcessedDataset_copy[:, 0].reshape(-1,1), ProcessedDataset_copy[:,1]
x_true, y_true = ProcessedDataset[:64, 0].reshape(-1, 1), ProcessedDataset[:64, 1]
x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(X_sample, y_sample, test_size=0.1, random_state=42)

#@C is 1/alpha and can be used to regulate the function
clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train_svm, y_train_svm)
y_true_pre_svm = clf.predict(x_true)
plt.plot(DateTime[:64], y_true_pre_svm[:], 'g--', linewidth=1, label="SVC model")

# RF Block
X_sample, y_sample = ProcessedDataset_copy[:, 0].reshape(-1,1), ProcessedDataset_copy[:,1]
x_true, y_true = ProcessedDataset[:, 0].reshape(-1, 1), ProcessedDataset[:, 1]
x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.25, random_state=19)
rf = RandomForestClassifier(n_estimators=1000, random_state=23)
rf.fit(x_train, y_train)
y_true_pre_rf = rf.predict(x_true)
plt.plot(DateTime[:64], y_true_pre_rf[:64], 'm--', linewidth=1, label="RF model")

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
plt.savefig("DAY1_overall_plots_test.png", format="png")
plt.close()