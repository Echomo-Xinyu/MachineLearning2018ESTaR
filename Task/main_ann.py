# The file is to build a model based on artificial neural network.
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=5)

result = []

while dt < end:
    result.append(float(dt.strftime('%Y%m%d%H%M%S')))
    dt += step

Time = np.asarray(result)

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
# print("CSR:\n", CSR)

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
    elif element[1] <= 1:
        element[2] = 10
    else:
        element[2] = 100000
ProcessedDataset = np.delete(ProcessedDataset, 1, 1)
np.random.shuffle(ProcessedDataset)
print("Shape of the processed dataset: ", ProcessedDataset.shape)

Features = ProcessedDataset[:, 0]
Labels = ProcessedDataset[:, 1]

def evaluate(model, test_features, test_labels, model_name):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print(model_name, ' Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.\n'.format(accuracy))
    return accuracy

# After running the best random state is 99, the score is only 49.99445289669128
train_features, test_features, train_labels, test_labels = train_test_split(Features, Labels, test_size=0.30, random_state=99)
train_features, test_features = train_features.reshape(-1, 1), test_features.reshape(-1, 1)

clf1 = MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 4), 
random_state=23, learning_rate='adaptive')
clf1.fit(train_features, train_labels)

clf2 = MLPClassifier(activation='logistic', solver='sgd', alpha=1e-4, hidden_layer_sizes=(5, 2), 
random_state=23, learning_rate='adaptive')
clf2.fit(train_features, train_labels)

clf3 = MLPClassifier(activation='tanh', solver='sgd', alpha=1e-5, hidden_layer_sizes=(20, 4), 
random_state=23, learning_rate='adaptive')
clf3.fit(train_features, train_labels)

clf4 = MLPClassifier(activation='logistic', solver='adam', alpha=1e-4, hidden_layer_sizes=(10, 4), 
random_state=23, learning_rate='adaptive', batch_size=200)
clf4.fit(train_features, train_labels)

ann1_accuracy = evaluate(clf1, test_features, test_labels, 'ann')
ann2_accuracy = evaluate(clf2, test_features, test_labels, 'ann')
ann3_accuracy = evaluate(clf3, test_features, test_labels, 'ann')
ann4_accuracy = evaluate(clf4, test_features, test_labels, 'ann')

end_time = datetime.datetime.now()
print("Time taken to run the program till complete the graph: ",
(end_time-start_time).seconds, " seconds")


