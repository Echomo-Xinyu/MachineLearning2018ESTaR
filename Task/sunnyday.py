import numpy as np
import datetime

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

RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.40')

SWDIR = RoughDataset[:, 1]
SWDIF = RoughDataset[:, 2]
GLW = RoughDataset[:, 3]

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
    elif element[1] <= 1.0:
        element[2] = 10
    else:
        element[2] = 0
# ProcessedDataset = np.delete(ProcessedDataset, 1, 1)

highest_mean_index_a, highest_mean_index_b = 0, 0
highest_day_mean = 0
day_index = 0
for j in range(365):
    indexa, indexb = j * 96, (j+1) * 96
    current_day_mean = np.mean(ProcessedDataset[indexa:indexb, 2])
    # ProcessedDataset[indexa:indexb, 3] = current_day_mean
    if highest_day_mean < current_day_mean:
        highest_day_mean = current_day_mean
        highest_mean_index_a, highest_mean_index_b = indexa, indexb
        day_index = j

print("Highest day mean is: ", highest_day_mean)
print("The corresponding index is: ", day_index, " ",highest_mean_index_a , " ", highest_mean_index_b)
        