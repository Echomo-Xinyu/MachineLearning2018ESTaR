# This file is to combine data from 63 stns and their correspond locations
# and apply svm model to it.
import numpy as np
import datetime
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

print("The program running now is the model based on svm.")
tslist_path = "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/tslist"
# 2944*62 = 182528
FinalDataset = np.zeros(shape=(182528, 8), dtype=float)
location_nparray = np.zeros(shape=(63, 3), dtype=float)
def ReadList(file_pathway):
    f = open(file_pathway)
    line_number = 1
    i = 0
    for data in f.readlines():
        if line_number == 1 or line_number == 2 or line_number == 3:
            line_number+=1
        else:
            # remove the \n and space between data
            data = data.strip('\n')
            nums = data.split(' ')
            while '' in nums:
                nums.remove('')
            string = nums[1]
            index = string[3:]
            location_nparray[i,0] = index
            location_nparray[i,1], location_nparray[i,2] = nums[2], nums[3]
            i += 1
            if i == 63:
                break
    f.close()
ReadList(tslist_path)

# Read the data
def ReadFile(file_pathway):
    f = open(file_pathway)
    first_else = True
    line_number = 1
    for data in f.readlines():
        if line_number==1:
            line_number+=1
            continue
        if line_number <= 2945:
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
        else:
            break
    matrix = matrix.transpose()
    f.close()
    return matrix

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2015, 1, 31, 23, 59, 59)
step = datetime.timedelta(minutes=15)
result1 = []
result2 = []
while dt < end:
    result1.append(float(dt.strftime('%m%d%H')))
    result2.append(float(dt.strftime('%m')))
    dt += step
Time_long = np.asarray(result1)
month_index = np.asarray(result2)

for k in range(62):
    index = k + 1
    file_path =  "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata." + str(index)
    location = location_nparray[index]
    startindex, endindex = 2944 * k, 2944 * (k+1)
    FinalDataset[startindex:endindex, :3] = location

    RoughDataset_1 = np.asarray(ReadFile(file_path))
    RoughDataset = RoughDataset_1[:2944, :]
    SWDIR = RoughDataset[:, 1]
    SWDIF = RoughDataset[:, 2]
    GLW = RoughDataset[:, 3]

    CSR = SWDIF / (SWDIF + SWDIR)
    dataSize = np.size(Time_long, 0)
    # print("CSR shape: ", CSR.shape)
    # print("Time_long shape: ", Time_long.shape)
    # print("month_index: ", month_index.shape)

    ProcessedDataset = np.vstack((Time_long, month_index, CSR))
    ProcessedDataset = ProcessedDataset.transpose()

    zeroVector = np.zeros(shape=(1, dataSize))
    # 3 for day average value, 4 for month average value, 5 for CSR grouping
    ProcessedDataset = np.insert(ProcessedDataset, 3, values=zeroVector, axis=1)
    ProcessedDataset = np.insert(ProcessedDataset, 4, values=zeroVector, axis=1)
    ProcessedDataset = np.insert(ProcessedDataset, 5, values=zeroVector, axis=1)
    # below is to convert ratio CSR to correspond group 0-10
    for element in ProcessedDataset:
        if np.isnan(element[2]):
            element[5] = 0
        elif element[2] < 0.1:
            element[5] = 1
        elif element[2] < 0.2:
            element[5] = 2
        elif element[2] < 0.3:
            element[5] = 3
        elif element[2] < 0.4:
            element[5] = 4
        elif element[2] < 0.5:
            element[5] = 5
        elif element[2] < 0.6:
            element[5] = 6
        elif element[2] < 0.7:
            element[5] = 7
        elif element[2] < 0.8:
            element[5] = 8
        elif element[2] < 0.9:
            element[5] = 9
        else:
            element[5] = 10
    # mean CSR value each day
    for j in range(31):
        indexa, indexb = j * 4, (j+1) * 4
        ProcessedDataset[indexa:indexb, 3] = np.mean(ProcessedDataset[indexa:indexb, 5])
    # # mean CSR value each mont
    # for i in range(12):
    #     MonthIndex = i + 1
    #     DayNumber1, DayNumber2 = 0, 0
    #     DayNumber1 = TotalDayInMonth(i)
    #     DayNumber2 = TotalDayInMonth(MonthIndex)
    #     if i == 11:
    #         ProcessedDataset[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset[DayNumber1:, 5])
    #         break
    #     ProcessedDataset[DayNumber1:DayNumber2, 4] = np.mean(ProcessedDataset[DayNumber1:DayNumber2, 5])
    ProcessedDataset[:, 4] = np.mean(ProcessedDataset[:, 5])
    # ProcessedDataset should contain 5 columns:
    # time long float, month index, day mean CSR, month mean CSR, actual CSR group
    ProcessedDataset = np.delete(ProcessedDataset, 2, 1)
    # print("Shape of ProcessedDataset: ", ProcessedDataset.shape)
    # print("Shape of fit in final dataset: ", FinalDataset[startindex:endindex, 3:8].shape)
    FinalDataset[startindex:endindex, 3:] = ProcessedDataset
    print("Current range index k is: ", k)

print("Great to know you have completed the read in of dataset!!!")
print("Now is the more time consuming part..")
# np.random.shuffle(FinalDataset)
x_train, y_train = FinalDataset[:, :7], FinalDataset[:, 7]
print("Shape of final dataset: ", FinalDataset.shape)

start_time = datetime.datetime.now()
#@C is 1/alpha and can be used to regulate the function
clf = svm.SVC(C=0.1, kernel='sigmoid', gamma='auto', decision_function_shape='ovr',
max_iter=1000000, random_state=23, coef0=1.0)
clf.fit(x_train, y_train)
print("Data fit in successfully!")

y_train_pre = clf.predict(x_train)
print("train: MAE: ", metrics.mean_absolute_error(y_train, y_train_pre))
print("train: MSE: ",  metrics.mean_squared_error(y_train, y_train_pre))
print("train: RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, y_train_pre)))

end_time = datetime.datetime.now()
print("Time taken to run the program till complete the first model: ", (end_time-start_time).seconds, " seconds")

print("Now plotting the predicted trend..")

plt.plot(Time_long[:], y_train_pre[:2944], 'b--', linewidth=1)
plt.plot(Time_long[:], y_train[:2944], 'r,')
plt.xlabel("Time")
plt.ylabel("CSR")
figureName = "SVM_Jan.svg"
plt.savefig(figureName, format='svg')
plt.close()