# The file is to compare between the svm model and random forest model
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
# import seaborn as sns
import datetime
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

start_time = datetime.datetime.now()
handle = open("csf_console_log.txt", "w")


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
handle.write("This is the start of the program.")
RoughDataset = ReadFile('wrfdata.5')
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
# print("CSR:\n", CSR)

dataSize = np.size(Time, 0)
print("total number of useful datas is: ", dataSize)
handle.write("Total number of useful datas is: " + str(dataSize))

ProcessedDataset = np.vstack((Time, CSR))
ProcessedDataset = ProcessedDataset.transpose()
# print("Processed dataset: \n", ProcessedDataset)
# print("Processed dataset's shape: ", ProcessedDataset.shape, "\n\n")

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
    handle.write(model_name + 'Performance:')
    handle.write('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    handle.write('Accuracy = {:0.2f}%.\n'.format(accuracy))
    return accuracy

overall_accuracy_difference = 0

for x in range(10):
    print("\nThis is the ", x, " run:")
    handle.write('This is the ' + str(x) + " BIG run: ")
    random_state_thisRun = x * 10 + 3
    print("The random state in this run is: ", random_state_thisRun)
    handle.write('The random state in this big run is ' + str(random_state_thisRun))
    ProcessedDataset = shuffle(ProcessedDataset, random_state = random_state_thisRun)
    # print("Shape of processed dataset: ", ProcessedDataset.shape)
    for i in range(10):
        # index number for the test data
        initial_number, ending_number = i * 5226, i * 5226 + 5226
        test_dataset = ProcessedDataset[initial_number:ending_number]
        # print("Shape of test dataset: ", test_dataset.shape)
        test_features, test_labels = test_dataset[:, 0], test_dataset[:, 1]
        train_dataset = np.delete(ProcessedDataset, slice(initial_number, ending_number), axis=0)
        # print("Shape of the train dataset: ", train_dataset.shape)
        train_features, train_labels = train_dataset[:, 0], train_dataset[:, 1]
        train_features, test_features = train_features.reshape(-1, 1), test_features.reshape(-1, 1)

        # print('Training Features Shape:', train_features.shape)
        # print('Training Labels Shape:', train_labels.shape)
        # print('Testing Features Shape:', test_features.shape)
        # print('Testing Labels Shape:', test_labels.shape, '\n')
        # print("Test labels: ", test_labels, '\n\n\n')

        clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovr')
        clf.fit(train_features, train_labels)
        svm_accuracy = evaluate(clf, test_features, test_labels, 'svm')

        rf_model = RandomForestClassifier(n_estimators = 500)
        rf_model.fit(train_features, train_labels)
        rf_accuracy = evaluate(rf_model, test_features, test_labels, 'rf')

        diff = svm_accuracy - rf_accuracy
        overall_accuracy_difference += diff
        print("The difference between the ", x, " time svm model and rf model is: ", diff, '\nPositive means svm has a higher accuracy and bvice versa.')
        print("Complete the ", i, " run in the ", x, " run.\n\n")
        handle.write('The difference between the ' + str(x) + ' time svm model and rf model is: ' + str(diff))

print("Overall accuracy difference: ", overall_accuracy_difference)
handle.write("Overall accuracy difference: " + str(overall_accuracy_difference))

handle.close()







