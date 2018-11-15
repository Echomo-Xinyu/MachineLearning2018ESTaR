# The file is for the neural netwrok model
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import datetime
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
# handle.write("This is the start of the program.")
RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.5')
# print("Data for the first day:\n", RoughDataset)

Time = RoughDataset[:,0]
SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

# count variable is to indicate how many number of 0 we've got so far
# count_line is to store the line number of zero result line
count = 0
count_line = list()

n = np.size(Time, 0)
for i in range(n):
    if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
        count+=1
        count_line.append(i)

# print("total number of removed data (night time): ",count)

# @X_zero array is to store all the set of values during night time
X_zero = np.zeros(shape=(count, 4))

count_line.sort()
for number in reversed(range(count)):
    line_number = count_line[number]
    X_zero[number][0] = Time[line_number]
    X_zero[number][1] = SWDIR[line_number]
    X_zero[number][2] = SWDIF[line_number]
    X_zero[number][3] = GLW[line_number]
    Time = np.delete(Time, line_number, 0)
    SWDIR = np.delete(SWDIR, line_number, 0)
    SWDIF = np.delete(SWDIF, line_number, 0)
    GLW = np.delete(GLW, line_number, 0)
CSR = SWDIF / (SWDIF + SWDIR)
dataSize = np.size(Time, 0)
print("total number of useful datas is: ", dataSize)
ProcessedDataset = np.vstack((Time, CSR))
ProcessedDataset = ProcessedDataset.transpose()

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
    else:
        element[2] = 10
ProcessedDataset = np.delete(ProcessedDataset, 1, 1)

np.random.shuffle(ProcessedDataset)
# print("Shape of the processed dataset: ", ProcessedDataset.shape)

def evaluate(model, test_features, test_labels, model_name):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print(model_name, ' Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.\n'.format(accuracy))
    return accuracy

overall_accuracy_difference = 0
highest_svm_accuracy = 0
highest_rf_accuracy = 0
best_svm_big_run = 0
best_svm_small_run = 0
best_rf_big_run = 0
best_rf_small_run = 0

# original random index is an iteration of many integers and generaye a relatively random integer
random_index = 1
ProcessedDataset = shuffle(ProcessedDataset, random_state = 33)
initial_number, ending_number = random_index * 5226, random_index * 5226 + 5226
test_dataset = ProcessedDataset[initial_number:ending_number]
test_features, test_labels = test_dataset[:, 0], test_dataset[:, 1]
train_dataset = np.delete(ProcessedDataset, slice(initial_number, ending_number), axis=0)
train_features, train_labels = train_dataset[:, 0], train_dataset[:, 1]
train_features, test_features = train_features.reshape(-1, 1), test_features.reshape(-1, 1)

MLP = MLPClassifier(activation='logistic', alpha=0, hidden_layer_sizes=(100000, 100), learning_rate='adaptive')
MLP.fit(train_features, train_labels)
test_accuracy = evaluate(MLP, test_features, test_labels, 'ann_test')
print("The test accuracy is: ", test_accuracy)
train_accuracy = evaluate(MLP, train_features, train_labels, 'ann_train')
print("The train accuracy is: ", train_accuracy)

def sdError(model, train_features, train_labels, test_features, test_labels):
    predictions_train = model.predict(train_features)
    train_error = np.mean(predictions_train - train_labels)
    predictions_test = model.predict(test_features)
    test_error = np.mean(predictions_test - test_labels)
    return train_error, test_error

train_error, test_error = sdError(MLP, train_features, train_labels, test_features, test_labels)
print("The self-defined train error is: ", train_error)
print("The self-defined test error is: ", test_error)
print("The self definition is just the mean value of difference")

Time = Time.reshape(-1, 1)
ann_predicted_CSR = MLP.predict(Time)
MonthName = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("Now plotting the predicted trend by svm and rf model as well as teh actual one..")

i_range = [0, 1]
for i in i_range:
    TimeInitialValue = i * 4355
    TimeFinalValue = i * 4355 + 4355
    CurrentMonth = MonthName[i]
    plt.plot(Time[TimeInitialValue:TimeFinalValue], ann_predicted_CSR[TimeInitialValue:TimeFinalValue], 'b--', linewidth=1)
    plt.plot(Time[TimeInitialValue:TimeFinalValue], CSR[TimeInitialValue:TimeFinalValue], 'r--', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('CSR(predicted)')
    figureName = CurrentMonth + 'ann' + '1.svg'
    plt.savefig(figureName, format="svg")
    plt.close()

MLP = MLPClassifier(activation='tanh', alpha=0, hidden_layer_sizes=(100000, 100), learning_rate='adaptive')
MLP.fit(train_features, train_labels)
test_accuracy = evaluate(MLP, test_features, test_labels, 'ann_test')
print("The test accuracy is: ", test_accuracy)
train_accuracy = evaluate(MLP, train_features, train_labels, 'ann_train')
print("The train accuracy is: ", train_accuracy)

train_error, test_error = sdError(MLP, train_features, train_labels, test_features, test_labels)
print("The self-defined train error is: ", train_error)
print("The self-defined test error is: ", test_error)
print("The self definition is just the mean value of difference")

Time = Time.reshape(-1, 1)
ann_predicted_CSR = MLP.predict(Time)
MonthName = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("Now plotting the predicted trend by svm and rf model as well as teh actual one..")

i_range = [0, 1]
for i in i_range:
    TimeInitialValue = i * 4355
    TimeFinalValue = i * 4355 + 4355
    CurrentMonth = MonthName[i]
    plt.plot(Time[TimeInitialValue:TimeFinalValue], ann_predicted_CSR[TimeInitialValue:TimeFinalValue], 'b--', linewidth=1)
    plt.plot(Time[TimeInitialValue:TimeFinalValue], CSR[TimeInitialValue:TimeFinalValue], 'r--', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('CSR(predicted)')
    figureName = CurrentMonth + 'ann' + '2.svg'
    plt.savefig(figureName, format="svg")
    plt.close()

MLP = MLPClassifier(activation='relu', alpha=0, hidden_layer_sizes=(100000, 100), learning_rate='adaptive')
MLP.fit(train_features, train_labels)
test_accuracy = evaluate(MLP, test_features, test_labels, 'ann_test')
print("The test accuracy is: ", test_accuracy)
train_accuracy = evaluate(MLP, train_features, train_labels, 'ann_train')
print("The train accuracy is: ", train_accuracy)

train_error, test_error = sdError(MLP, train_features, train_labels, test_features, test_labels)
print("The self-defined train error is: ", train_error)
print("The self-defined test error is: ", test_error)
print("The self definition is just the mean value of difference")

Time = Time.reshape(-1, 1)
ann_predicted_CSR = MLP.predict(Time)
MonthName = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("Now plotting the predicted trend by svm and rf model as well as teh actual one..")

i_range = [0, 1]
for i in i_range:
    TimeInitialValue = i * 4355
    TimeFinalValue = i * 4355 + 4355
    CurrentMonth = MonthName[i]
    plt.plot(Time[TimeInitialValue:TimeFinalValue], ann_predicted_CSR[TimeInitialValue:TimeFinalValue], 'b--', linewidth=1)
    plt.plot(Time[TimeInitialValue:TimeFinalValue], CSR[TimeInitialValue:TimeFinalValue], 'r--', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('CSR(predicted)')
    figureName = CurrentMonth + 'ann' + '3.svg'
    plt.savefig(figureName, format="svg")
    plt.close()

end_time = datetime.datetime.now()
print("Time taken to run the program till complete the graph: ", (end_time-start_time).seconds, " seconds")
