import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
# import seaborn as sns
import datetime
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates

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

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=15)
result = []
result2 = []
while dt < end:
    result.append(float(dt.strftime('%m%d%H%M')))
    result2.append(dt.strftime("%m/%d/%H:%M"))
    dt += step
Time = np.asarray(result)
DateTime = np.asarray(result2)

RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.55')
print("Data for the first day:\n", RoughDataset)


SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

CSR = SWDIF / (SWDIF + SWDIR)
print("CSR:\n", CSR)

dataSize = np.size(Time, 0)
print("total number of useful datas is: ", dataSize)

dataSize = np.size(Time, 0)
print("total number of useful datas is: ", dataSize)

ProcessedDataset = np.vstack((Time, CSR))
ProcessedDataset = ProcessedDataset.transpose()
print("Processed dataset: \n", ProcessedDataset)
print("Processed dataset's shape: ", ProcessedDataset.shape, "\n")

zeroVector = np.zeros(shape=(1, dataSize))
ProcessedDataset = np.insert(ProcessedDataset, 2, values=zeroVector, axis=1)

for element in ProcessedDataset:
    if np.isnan(element[1]):
        element[2] = 0
    elif element[1] < 0.1:
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
ProcessedDataset_copy = ProcessedDataset.copy()

np.random.shuffle(ProcessedDataset_copy)
print("Shape of the processed dataset: ", ProcessedDataset.shape)

X_sample, y_sample = ProcessedDataset_copy[:, 0].reshape(-1,1), ProcessedDataset_copy[:,1]
x_true, y_true = ProcessedDataset[:, 0].reshape(-1, 1), ProcessedDataset[:, 1]
x_train, x_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=1000, random_state=23)
rf.fit(x_train, y_train)

# y_train_pre = rf.predict(x_train)
# print("train: MAE: ", metrics.mean_absolute_error(y_train, y_train_pre))
# print("train: MSE: ",  metrics.mean_squared_error(y_train, y_train_pre))
# print("train: RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, y_train_pre)))

# y_test_pre = rf.predict(x_test)
# print("train: MAE: ", metrics.mean_absolute_error(y_test, y_test_pre))
# print("train: MSE: ",  metrics.mean_squared_error(y_test, y_test_pre))
# print("train: RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_test_pre)))

# # train: MAE:  0.0
# # train: MSE:  0.0
# # train: RMSE:  0.0
# # train: MAE:  0.7460045662100456
# # train: MSE:  4.425799086757991
# # train: RMSE:  2.1037583242278544


# MonthName = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# print("Now plotting the predicted trend..")

# # for i in range(12):
# #     TimeInitialValue = i * 2920
# #     TimeFinalValue = i * 2920 + 2920
# #     CurrentMonth = MonthName[i]
# #     plt.plot(Time[TimeInitialValue:TimeFinalValue], y_true_pre[TimeInitialValue:TimeFinalValue], 'b--', linewidth=1)
# #     plt.plot(Time[TimeInitialValue:TimeFinalValue], y_true[TimeInitialValue:TimeFinalValue], 'r,')
# #     plt.xlabel('Time')
# #     plt.ylabel('CSR(predicted)')
# #     figureName = "RF_O_" + CurrentMonth + '.svg'
# #     plt.savefig(figureName, format="svg")
# #     plt.close()
y_true_pre = rf.predict(x_true)
# print("true: MAE: ", metrics.mean_absolute_error(y_true, y_true_pre))
# print("true: MSE: ",  metrics.mean_squared_error(y_true, y_true_pre))
# print("true: RMSE: ", np.sqrt(metrics.mean_squared_error(y_true, y_true_pre)))

# plt.plot(Time[:64], y_true_pre[:64], 'b--', linewidth=1)
# plt.plot(Time[:64], y_true[:64], 'r,')
# plt.xlabel("Time")
# plt.ylabel("CSR by RF")
# plt.savefig("RF_DAY1.svg", forma='svg')
# plt.close()
# print(y_true_pre)
# print(x_true)

Dt_formatted = [datetime.datetime.strptime(d, "%m/%d/%H:%M").date() for d in DateTime]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

plt.plot(Dt_formatted[:], y_true_pre[:], 'b--', linewidth=1)
plt.plot(Dt_formatted[:], y_true[:], 'r,')
plt.xlabel('Time(MM/DD)')
plt.ylabel('CSR by RF')
plt.title("")
plt.savefig("RF_Overall.svg", format="svg")
plt.close()