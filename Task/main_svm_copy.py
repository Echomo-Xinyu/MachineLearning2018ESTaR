import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
# import seaborn as sns
import datetime
from sklearn import svm
from sklearn import metrics

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
RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.5')
SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=15)
result1 = []
result2 = []
while dt < end:
    result1.append(float(dt.strftime('%Y%m%d%H%M%S')))
    result2.append(float(dt.strftime('%m')))
    dt += step
Time_long = np.asarray(result1)
month_index = np.asarray(result2)

CSR = SWDIF / (SWDIF + SWDIR)
print("CSR:\n", CSR)

dataSize = np.size(Time_long, 0)
print("total number of useful datas is: ", dataSize)

ProcessedDataset = np.vstack((Time_long, month_index, CSR))
ProcessedDataset = ProcessedDataset.transpose()
print("Processed dataset: \n", ProcessedDataset)
print("Processed dataset's shape: ", ProcessedDataset.shape, "\n\n")

zeroVector = np.zeros(shape=(1, dataSize))
# 3 for day average value, 4 for month average value, 5 for CSR grouping
ProcessedDataset = np.insert(ProcessedDataset, 3, values=zeroVector, axis=1)
ProcessedDataset = np.insert(ProcessedDataset, 4, values=zeroVector, axis=1)
ProcessedDataset = np.insert(ProcessedDataset, 5, values=zeroVector, axis=1)

# below is to convert ratio CSR to correspond group 0-10
for element in ProcessedDataset:
    if np.isnan(element[2]):
        element[5] = 0
    if element[2] < 0.1:
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
for i in range(365):
    indexa, indexb = i * 96, (i+1) * 96
    ProcessedDataset[indexa:indexb, 3] = np.mean(ProcessedDataset[indexa:indexb, 5])

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

for i in range(12):
    MonthIndex = i + 1
    DayNumber1, DayNumber2 = 0, 0
    DayNumber1 = TotalDayInMonth(i)
    DayNumber2 = TotalDayInMonth(MonthIndex)
    if i == 11:
        ProcessedDataset[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset[DayNumber1:, 5])
        break
    ProcessedDataset[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset[DayNumber1:DayNumber2, 5])


ProcessedDataset = np.delete(ProcessedDataset, 2, 1)

# The function is to calculate the mean absolute error
def MAE(obtainedValue, actualValue):
    error = np.abs(obtainedValue - actualValue)
    return np.mean(error)

def MSE(obtainedValue, actualValue):
    error = np.abs(obtainedValue - actualValue)
    return np.sum(error ** 2)

np.random.shuffle(ProcessedDataset)
print("Shape of the processed dataset: ", ProcessedDataset.shape)

trainingSet = ProcessedDataset[:, :]
# crossvalidationSet = ProcessedDataset[36586:47040, :]
testSet = ProcessedDataset[73584:, :]
print("testSet: ", testSet, "\n")
print("Shape of the testSet: ", testSet.shape)

x_train, y_train = trainingSet[:, 0:4], trainingSet[:, 4]
x_test, y_test = testSet[:, 0:4], testSet[:, 4]
# x_train, x_test = x_train.reshape(-1, 1), x_test.reshape(-1, 1)

#@C is 1/alpha and can be used to regulate the function
clf = svm.SVC(C=0.1, kernel='sigmoid', gamma='auto', decision_function_shape='ovr',
max_iter=100000000, random_state=23, coef0=1.0)
clf.fit(x_train, y_train)
y_train_pre = clf.predict(x_train)
y_test_pre = clf.predict(x_test)
print("train: MAE: ", metrics.mean_absolute_error(y_train_pre, y_train))
print("train: MSE: ",  metrics.mean_squared_error(y_train_pre, y_train))
print("test: MAE: ", metrics.mean_absolute_error(y_test_pre, y_test))
print("test: MSE: ", metrics.mean_squared_error(y_test_pre, y_test))
MAE(clf.predict(x_train), y_train)
MSE(clf.predict(x_train), y_train)
clf.score(x_train, y_train)
MAE(clf.predict(x_test), y_test)
MSE(clf.predict(x_test), y_test)
print()

print("clf train score: ", clf.score(x_train, y_train))

y_hat = clf.predict(x_train)
print("y_hat's shape: ", y_hat.shape)
# show_accuracy(y_hat, y_test, 'training set')
print("clf test score", clf.score(x_test, y_test))
# y_hat = clf.predict(x_test)
# show_accuracy(y_hat, y_test, 'test set')



def show_accuracy(obtainedValue, actualValue, Name):
    if obtainedValue.shape != actualValue.shape:
        print("The obtained value is not exactly the same shape as the actual values.\nComparison cannot be carried out.")
        return
    totalNumber = np.size(obtainedValue, 0)
    correctPrediction = 0
    for index in range(obtainedValue):
        if obtainedValue[index] == actualValue[index]:
            correctPrediction += 1
    MAEerror = MAE(obtainedValue, actualValue)
    print("The mean absolute error is: ", MAEerror)
    MSEerror = MSE(obtainedValue, actualValue)
    print("The mean square error is: ", MSEerror)
    print("The accuracy for ", Name, "is: ", correctPrediction/totalNumber)
    # return correctPrediction / totalNumber

show_accuracy(y_hat, y_test, "SVM")
# Time_long = Time_long.reshape(-1, 1)
# predicted_CSR = clf.predict(Time_long)
# MonthName = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# print("Now plotting the predicted trend..")

# for i in range(12):
#     TimeInitialValue = i * 4355
#     TimeFinalValue = i * 4355 + 4355
#     CurrentMonth = MonthName[i]
#     plt.plot(Time[TimeInitialValue:TimeFinalValue], predicted_CSR[TimeInitialValue:TimeFinalValue], 'b--', linewidth=1)
#     plt.xlabel('Time')
#     plt.ylabel('CSR(predicted)')
#     figureName = CurrentMonth + '.svg'
#     plt.savefig(figureName, format="svg")
#     plt.close()

