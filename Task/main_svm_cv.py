# This script is to test the function of different C values in svc 
# and the result is no.. So in short, the C value in defining the function cannot regularize the model.
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
RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Task/wrfdata.5')

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

result = []

while dt < end:
    result.append(float(dt.strftime('%Y%m%d%H%M%S')))
    dt += step

Time = np.asarray(result)

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

C_list = [0.01, 0.1, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.8, 2.0, 3.0, 4.0, 5.0]
Train_error_list = [[None] for n in range(len(C_list))]
Test_error_list = [[None] for n in range(len(C_list))]

def errorCV(model, trainingSet, testSet):
    x_train, y_train = trainingSet[:, 0], trainingSet[:, 1]
    x_test, y_test = testSet[:, 0], testSet[:,1]
    x_train, x_test = x_train.reshape(-1, 1), x_test.reshape(-1, 1)
    model.fit(x_train, y_train)
    predicted_y_train = model.predict(x_train)
    train_error = np.mean(np.abs(predicted_y_train - y_train))
    predicted_y_test = model.predict(x_test)
    test_error = np.mean(np.abs(predicted_y_test - y_test))
    return train_error, test_error

for element in C_list:
    c_value = element
    index = C_list.index(element)
    # @C is 1/alpha and can be used to regulate the function
    svm_model = svm.SVC(C=c_value, kernel='rbf', gamma=20, decision_function_shape='ovr')
    train_error, test_error = errorCV(svm_model, trainingSet, testSet)
    Train_error_list[index], Test_error_list[index] = train_error, test_error


plt.plot(C_list, Train_error_list, 'b-', C_list, Test_error_list, 'r-')
plt.title("SVM CV plot")
plt.xlabel("C value")
plt.ylabel("Abs from actual value")
plt.savefig("SVMCVplot.png", format="png")


