import numpy as np
import datetime
import math
import warnings
import itertools
import pandas as pd
# from pandas import datetime
import statsmodels.api as sm
from sklearn import svm
from sklearn import metrics
# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from statsmodels.tsa.arima_model import ARMA
# from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

startindex = "2015-01-01 8:00"
endindex = "2016-01-01 7:59"

TimeDF = pd.date_range(start=startindex, end=endindex, freq="15T")

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=15)
result1 = []
while dt < end:
    result1.append(float(dt.strftime('%m%d%H%M')))
    dt += step
Time_long = np.asarray(result1)

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
k = 54
print("Current station index k is: ", k)
index = k + 1
file_path =  "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata." + str(index)

RoughDataset = np.asarray(ReadFile(file_path))
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
    
ProcessedDaraser = np.vstack((Time, CSRGroup))

ProcessedDataset_copy = ProcessedDataset.copy()
np.random.shuffle(ProcessedDataset_copy)
print("Shape of the processed dataset: ", ProcessedDataset.shape)
X_sample, y_sample = ProcessedDataset_copy[:, 0].reshape(-1,1), ProcessedDataset_copy[:,1]
x_true, y_true = ProcessedDataset[:, 0].reshape(-1, 1), ProcessedDataset[:, 1]
x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.1, random_state=42)

#@C is 1/alpha and can be used to regulate the function
clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train)
y_true = clf.predict(x_true)

time_series = pd.DataFrame(CSRGroup, index=TimeDF)

Dt_formatted = [datetime.datetime.strptime(d, "%m/%d/%H:%M").date() for d in DateTime]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

plt.plot(Dt_formatted[:], y_true_pre[:], 'b--', linewidth=1)
plt.plot(Dt_formatted[:], y_true[:], 'r,')
plt.xlabel('Time')
plt.ylabel('CSR by SVM')
plt.savefig("SVM_overall.svg", format="svg")
plt.close()