import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
# import seaborn as sns
import datetime
from sklearn import svm


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
RoughDataset = ReadFile('wrfdata.5')

FirstDayData = RoughDataset[:,:]
print("Data for the first day:\n",FirstDayData)

SWDIR = FirstDayData[:, 1]
SWDIF = FirstDayData[:, 2]
GLW = FirstDayData[:, 3]

# count variable is to indicate how many number of 0 we've got so far
# count_line is to store the line number of zero result line
count = 0
count_line = list()

n = np.size(SWDIF, 0)
for i in range(n):
    if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
        count+=1
        count_line.append(i)

print("total number of removed data (night time): ",count)

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=5)

result1 = []
result2 = []

while dt < end:
    result1.append(float(dt.strftime('%Y%m%d%H%M%S')))
    result2.append(float(dt.strftime('%m')))
    dt += step

Time_long = np.asarray(result1)
month_index = np.asarray(result2)


# @X_zero array is to store all the set of values during night Time
X_zero = np.zeros(shape=(count, 4))

count_line.sort()
for number in reversed(range(count)):
    # if number > 0:
    #     number -= 1
    # try:
    line_number = count_line[number]
    X_zero[number][0] = Time_long[line_number]
    X_zero[number][1] = SWDIR[line_number]
    X_zero[number][2] = SWDIF[line_number]
    X_zero[number][3] = GLW[line_number]
    Time_long = np.delete(Time_long, line_number, 0)
    month_index = np.delete(month_index, line_number, 0)
    SWDIR = np.delete(SWDIR, line_number, 0)
    SWDIF = np.delete(SWDIF, line_number, 0)
    GLW = np.delete(GLW, line_number, 0)
    # except:
    #     print("number: ", number)
    #     print("line_number: ", line_number)
    #     print("X_zero.shape: ", X_zero.shape)


CSR = SWDIF / (SWDIF + SWDIR)
print("CSR:\n", CSR)

dataSize = np.size(Time_long, 0)
print("total number of useful datas is: ", dataSize)

ProcessedDataset = np.vstack((Time_long, month_index, CSR))
ProcessedDataset = ProcessedDataset.transpose()
print("Processed dataset: \n", ProcessedDataset)
print("Processed dataset's shape: ", ProcessedDataset.shape, "\n\n")

zeroVector = np.zeros(shape=(1, dataSize))
ProcessedDataset = np.insert(ProcessedDataset, 3, values=zeroVector, axis=1)

for element in ProcessedDataset:
    if element[2] < 0.1:
        element[3] = 0
    elif element[2] < 0.2:
        element[3] = 1
    elif element[2] < 0.3:
        element[3] = 2
    elif element[2] < 0.4:
        element[3] = 3
    elif element[2] < 0.5:
        element[3] = 4
    elif element[2] < 0.6:
        element[3] = 5
    elif element[2] < 0.7:
        element[3] = 6
    elif element[2] < 0.8:
        element[3] = 7
    elif element[2] < 0.9:
        element[3] = 8
    else:
        element[3] = 9
ProcessedDataset = np.delete(ProcessedDataset, 2, 1)

np.random.shuffle(ProcessedDataset)
print("Shape of the processed dataset: ", ProcessedDataset.shape)

trainingSet = ProcessedDataset[:47040, :]
# crossvalidationSet = ProcessedDataset[36586:47040, :]
testSet = ProcessedDataset[47040:, :]
print("testSet: ", testSet, "\n")
print("Shape of the testSet: ", testSet.shape)

x_train, y_train = trainingSet[:, 0:2], trainingSet[:, 2]
x_test, y_test = testSet[:, 0:2], testSet[:,2]
x_train, x_test = x_train.reshape(-1, 1), x_test.reshape(-1, 1)

#@C is 1/alpha and can be used to regulate the function
clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train)

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

