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
RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.5')

FirstDayData = RoughDataset[:,:]
print("Data for the first day:\n",FirstDayData)

Time = FirstDayData[:,0]
SWDIR = FirstDayData[:, 1]
SWDIF = FirstDayData[:, 2]
GLW = FirstDayData[:, 3]

# count variable is to indicate how many number of 0 we've got so far
# count_line is to store the line number of zero result line
count = 0
count_line = list()

n = np.size(Time, 0)
for i in range(n):
    if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
        count+=1
        count_line.append(i)

print("total number of removed data (night time): ",count)

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
        element[2] = 0
    elif element[1] < 0.2:
        element[2] = 1
    elif element[1] < 0.3:
        element[2] = 2
    elif element[1] < 0.4:
        element[2] = 3
    elif element[1] < 0.5:
        element[2] = 4
    elif element[1] < 0.6:
        element[2] = 5
    elif element[1] < 0.7:
        element[2] = 6
    elif element[1] < 0.8:
        element[2] = 7
    elif element[1] < 0.9:
        element[2] = 8
    else:
        element[2] = 9
ProcessedDataset = np.delete(ProcessedDataset, 1, 1)

np.random.shuffle(ProcessedDataset)
print("Shape of the processed dataset: ", ProcessedDataset.shape)

trainingSet = ProcessedDataset[:47040, :]
# crossvalidationSet = ProcessedDataset[36586:47040, :]
testSet = ProcessedDataset[47040:, :]
print("testSet: ", testSet, "\n")
print("Shape of the testSet: ", testSet.shape)

x_train, y_train = trainingSet[:, 0], trainingSet[:, 1]
x_test, y_test = testSet[:, 0], testSet[:,1]
x_train, x_test = x_train.reshape(-1, 1), x_test.reshape(-1, 1)

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

Time = Time.reshape(-1, 1)
predicted_CSR = clf.predict(Time)
MonthName = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("Now plotting the predicted trend..")

for i in range(12):
    TimeInitialValue = i * 4355
    TimeFinalValue = i * 4355 + 4355
    CurrentMonth = MonthName[i]
    plt.plot(Time[TimeInitialValue:TimeFinalValue], predicted_CSR[TimeInitialValue:TimeFinalValue], 'b--', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('CSR(predicted)')
    figureName = CurrentMonth + '.svg'
    plt.savefig(figureName, format="svg")
    plt.close()









# def plotGraph(x_para, y_para):
#     print("Now the computer is drawing the graph. This may take a bit long time. Please wait patiently and take your dinner if you haven't.:)")
#     # popt, pcov = curve_fit(gaussian, x_para, y_para, p0=[np.max(y_para), np.median(x_para), np.std(x_para), np.min(y_para)])
#     popt, pcov = curve_fit(gaussian, x_para, y_para)
#     # plot original data
#     plt.plot(x_para, y_para, 'b,', label='data')
#     # plot fit function
#     x_fit = np.linspace(np.min(x_para), np.max(x_para), 1000)
#     plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label='fit')
#     plt.legend()
#     plt.title('Fig for CSR against Time on the first day')
#     plt.xlabel('Time')
#     plt.ylabel('CSR')

#     end_time = datetime.datetime.now()
#     print("Time taken to run the program till complete the graph: ", (end_time-start_time).seconds, " seconds")

#     plt.show()