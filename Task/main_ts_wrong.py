# The file is to contain the time series model
# However, I fail to convert the given time series to the correspond time value,
# and find it would be eaiser to simply create a new vector of time series
# Refer to main_ts.py for the machine generated datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

Time = RoughDataset[:,0]
TotalValueOfTime = np.size(Time, 0)
ZeroIndexVector = np.zeros(shape=(TotalValueOfTime, ))
print("Shape of ZeroIndexVector: ", ZeroIndexVector.shape)
print("Shape of the Time: ", Time.shape)
IndexedTime = np.vstack((ZeroIndexVector, ZeroIndexVector, ZeroIndexVector, ZeroIndexVector, ZeroIndexVector, Time))
ZeroIndexNPArray = np.zeros(shape=(96, 5))

IndexedTime = IndexedTime.transpose()
print("Shape of the Indexed Time: ", IndexedTime.shape)
for i in range(TotalValueOfTime):
    indexNumber = i + 1
    # month, day, hour, minute respectively
    IndexedTime[i, 0] = indexNumber
    IndexedTime[i, 1] = indexNumber
    IndexedTime[i, 2] = indexNumber
    IndexedTime[i, 3] = indexNumber
    IndexedTime[i, 4] = indexNumber
print(IndexedTime)
print("Now changing the first two columns to the month and day value..")
NumberofDaysInYear = 365
RespDayLength = TotalValueOfTime / NumberofDaysInYear
IndexedTime[:, 1], IndexedTime[:, 2] = np.floor(IndexedTime[:, 1]/RespDayLength + 1), np.floor(IndexedTime[:, 2]/RespDayLength + 1)
lastValue = 0
# ExceptionLineNumber = list()
MonthNumber = 0
CorrepDayValueMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

for i in range(TotalValueOfTime+96):
    if i < TotalValueOfTime:
        IndexedTime[i, 0] = 2015
    else:
        IndexedTime[i, 0] = 2016

# @check<onth is to decide which month the nth day belong to
def checkMonth(DayValue):
    if (DayValue <= sum(CorrepDayValueMonth[0:1])):
        return 1
    elif (DayValue <= sum(CorrepDayValueMonth[0:2])):
        return 2
    elif (DayValue <= sum(CorrepDayValueMonth[0:3])):
        return 3
    elif (DayValue <= sum(CorrepDayValueMonth[0:4])):
        return 4
    elif (DayValue <= sum(CorrepDayValueMonth[0:5])):
        return 5
    elif (DayValue <= sum(CorrepDayValueMonth[0:6])):
        return 6
    elif (DayValue <= sum(CorrepDayValueMonth[0:7])):
        return 7
    elif (DayValue <= sum(CorrepDayValueMonth[0:8])):
        return 8
    elif (DayValue <= sum(CorrepDayValueMonth[0:9])):
        return 9
    elif (DayValue <= sum(CorrepDayValueMonth[0:10])):
        return 10
    elif (DayValue <= sum(CorrepDayValueMonth[0:11])):
        return 11
    elif (DayValue <= sum(CorrepDayValueMonth[:])):
        return 12
    else:
        print("This is a strange input check Month value. Please check again.")

# @removeExtraDayValue is to change the nth day of the year to the nth day of the month
def removeExtraDayValue(MonthValue, DayValue):
    if MonthValue == 1:
        return DayValue
    elif MonthValue == 2:
        return DayValue - sum(CorrepDayValueMonth[0:1])
    elif MonthValue == 3:
        return DayValue - sum(CorrepDayValueMonth[0:2])
    elif MonthValue == 4:
        return DayValue - sum(CorrepDayValueMonth[0:3])
    elif MonthValue == 5:
        return DayValue - sum(CorrepDayValueMonth[0:4])
    elif MonthValue == 6:
        return DayValue - sum(CorrepDayValueMonth[0:5])
    elif MonthValue == 7:
        return DayValue - sum(CorrepDayValueMonth[0:6])
    elif MonthValue == 8:
        return DayValue - sum(CorrepDayValueMonth[0:7])
    elif MonthValue == 9:
        return DayValue - sum(CorrepDayValueMonth[0:8])
    elif MonthValue == 10:
        return DayValue - sum(CorrepDayValueMonth[0:9])
    elif MonthValue == 11:
        return DayValue - sum(CorrepDayValueMonth[0:10])
    elif MonthValue == 12:
        return DayValue - sum(CorrepDayValueMonth[0:11])
    else:
        # print("This is an unexpected tuple of input. Please check again")
        return 0
    
for i in range(TotalValueOfTime):
    currentDayValue = IndexedTime[i, 1]
    if currentDayValue != lastValue and lastValue != 0:
        exceptionLine = i
        # ExceptionLineNumber.append(exceptionLine)
        IndexedTime[exceptionLine, 1] -= 1
        IndexedTime[exceptionLine, 2] -= 1
        lastValue = 0
    elif (lastValue == 0):
        lastValue = currentDayValue
    # IndexedTime[i, 0] = checkMonth(currentDayValue)
    # IndexedTime[i, 1] = removeExtraDayValue(IndexedTime[i, 0], currentDayValue)

# print(ExceptionLineNumber)
# for line_number in ExceptionLineNumber:

for i in range(TotalValueOfTime):
    DayValue = IndexedTime[i, 2]
    # print("DayValue: ", DayValue)
    MonthValue = checkMonth(DayValue)
    IndexedTime[i, 1] = MonthValue
    IndexedTime[i, 2] = removeExtraDayValue(MonthValue, DayValue)
    # print("The line number: ", i)
    # print("The month value: ", IndexedTime[i, 0])
    # print("The day value: ", IndexedTime[i, 1])

print(IndexedTime.shape)
print("Processed IndexedTime 1: \n", IndexedTime)
'''
so far I've made firts two columns of IndexedTime to be the Month Value and Day Value.
'''
print("Now changing the third and fourth value to be the hour and minute value..")
# 
for i in range(TotalValueOfTime):
    MonthValue = IndexedTime[i, 1]
    DayValue = IndexedTime[i, 2]
    OverallDayNumber = math.floor(IndexedTime[i, 3]/RespDayLength)
    
    IndexedTime[i, 3], IndexedTime[i, 4] = IndexedTime[i, 4] - OverallDayNumber * 288, IndexedTime[i, 4] - OverallDayNumber * 288
    if IndexedTime[i, 3] > 288 or IndexedTime[i, 3] < 0:
        print("Error. Line number is: ", i)
    if IndexedTime[i, 4] > 288 or IndexedTime[i, 4] < 0:
        print("Error. Line number is: ", i)

# print("Line 0: ", IndexedTime[0, :])
# print("Line 1550: ", IndexedTime[1550, :])

NumberofHourInDay = 24
IndexedTime[:, 3] = np.floor(IndexedTime[:, 3] / NumberofHourInDay)
# for element in IndexedTime[:, 2]:
#     if element > 24 or element < 0:
#         print("Error. The element is: ", element)

# TODOIST: change the ending time to 


print("\n\nThe most update indexedTime is: ", IndexedTime)

# SWDIR = RoughDataset[:, 1]
# SWDIF = RoughDataset[:, 2]
# GLW = RoughDataset[:, 3]

# # count variable is to indicate how many number of 0 we've got so far
# # count_line is to store the line number of zero result line
# count = 0
# count_line = list()

# n = np.size(Time, 0)
# for i in range(n):
#     if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
#         count+=1
#         count_line.append(i)

# # print("total number of removed data (night time): ",count)

# # @X_zero array is to store all the set of values during night time
# X_zero = np.zeros(shape=(count, 4))

# count_line.sort()
# for number in reversed(range(count)):
#     # if number > 0:
#     #     number -= 1
#     # try:
#     line_number = count_line[number]
#     X_zero[number][0] = Time[line_number]
#     X_zero[number][1] = SWDIR[line_number]
#     X_zero[number][2] = SWDIF[line_number]
#     X_zero[number][3] = GLW[line_number]
#     Time = np.delete(Time, line_number, 0)
#     SWDIR = np.delete(SWDIR, line_number, 0)
#     SWDIF = np.delete(SWDIF, line_number, 0)
#     GLW = np.delete(GLW, line_number, 0)
#     # except:
#     #     print("number: ", number)
#     #     print("line_number: ", line_number)
#     #     print("X_zero.shape: ", X_zero.shape)


# CSR = SWDIF / (SWDIF + SWDIR)
# # print("CSR:\n", CSR)

# dataSize = np.size(Time, 0)
# print("total number of useful datas is: ", dataSize)

# ProcessedDataset = np.vstack((Time, CSR))
# ProcessedDataset = ProcessedDataset.transpose()
# # print("Processed dataset: \n", ProcessedDataset)
# # print("Processed dataset's shape: ", ProcessedDataset.shape, "\n\n")

# zeroVector = np.zeros(shape=(1, dataSize))
# ProcessedDataset = np.insert(ProcessedDataset, 2, values=zeroVector, axis=1)

# for element in ProcessedDataset:
#     if element[1] < 0.1:
#         element[2] = 1
#     elif element[1] < 0.2:
#         element[2] = 2
#     elif element[1] < 0.3:
#         element[2] = 3
#     elif element[1] < 0.4:
#         element[2] = 4
#     elif element[1] < 0.5:
#         element[2] = 5
#     elif element[1] < 0.6:
#         element[2] = 6
#     elif element[1] < 0.7:
#         element[2] = 7
#     elif element[1] < 0.8:
#         element[2] = 8
#     elif element[1] < 0.9:
#         element[2] = 9
#     else:
#         element[2] = 10
# ProcessedDataset = np.delete(ProcessedDataset, 1, 1)

# # np.random.shuffle(ProcessedDataset)
# print("Shape of the processed dataset: ", ProcessedDataset.shape)

# def evaluate(model, test_features, test_labels, model_name):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print(model_name, ' Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.\n'.format(accuracy))
#     return accuracy



end_time = datetime.datetime.now()
print("Time taken to run the program till complete the graph: ", (end_time-start_time).seconds, " seconds")
