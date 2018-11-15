import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
# import seaborn as sns
import datetime
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
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
print("Data for the first day:\n", RoughDataset)

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
    else:
        element[2] = 10
ProcessedDataset = np.delete(ProcessedDataset, 1, 1)

np.random.shuffle(ProcessedDataset)
print("Shape of the processed dataset: ", ProcessedDataset.shape)

# features store the training set of data
features = ProcessedDataset[:, 0]
# @labels store the value we want to predict
labels = ProcessedDataset[:, 1]

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 23)
train_features, test_features = train_features.reshape(-1, 1), test_features.reshape(-1, 1)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape, '\n')
print("Test labels: ", test_labels, '\n\n\n')

rf = RandomForestClassifier(n_estimators=1000, random_state=23)

clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(train_features, train_labels)
print("clf train score: ", clf.score(train_features, train_labels))

# # The following part is a cross-validation to find the optimal parameter for the random forest
# # though it doesnot really work..
# from pprint import pprint
# # Look at parameters used by our current forest
# print('Parameters currently in use:\n')
# pprint(rf.get_params())

# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(200, 2000, num = 10)]
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10, 25]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4, 8]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# pprint(random_grid)

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=23, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(train_features, train_labels)
# pprint(rf_random.get_params())

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.\n'.format(accuracy))
    
    return accuracy

base_model = RandomForestClassifier(n_estimators = 1000, random_state = 23)
base_model.fit(train_features, train_labels)
# base_accuracy = evaluate(base_model, test_features, test_labels)
print('rf score is: ', base_model.score(train_features, train_labels))

# best_random = rf_random.best_estimator_
# random_accuracy = evaluate(best_random, test_features, test_labels)

# print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

'''
# rf.fit(train_features, train_labels)

# # Use the forest's predict method on the test data
# predictions = rf.predict(test_features)
# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.\n\n')

# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / test_labels)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
'''

# # Draw a decision tree
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = rf.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')