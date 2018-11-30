# The file is planned to find the correlation of solar radiation between different places
import numpy as np

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

# In the following case, I have included NaN inside the processed dataset as the nan value in CSR
# may not be the same number at different locations, and np.correlate requires an input of ????

# - [X] try different input-array mode
# - [ ] try both nan-clean input array 
# - [ ] try both nan-present input array



TotalDataset = np.asarray([[[0 for i in range(4)] for j in range(35040)] for k in range(62)])
ProcessedDataset = np.asarray([[[0 for i in range(2)] for j in range(35040)] for k in range(62)])
for i in range(62):
    index = str(i+1)
    pathway = "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata." + index
    TotalDataset[i] = np.asarray(ReadFile(pathway))
    Time = TotalDataset[i, :, 0]
    SWDIR = TotalDataset[i, :, 1]
    SWDIF = TotalDataset[i, :, 2]
    GLW = TotalDataset[i, :, 3]
    # count = 0
    # count_line = list()
    # n = np.size(Time, 0)
    # for i in range(n):
    #     if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
    #         count+=1
    #         count_line.append(i)
    # count_line.sort()
    # for number in reversed(range(count)):
    #     line_number = count_line[number]
    #     Time = np.delete(Time, line_number, 0)
    #     SWDIR = np.delete(SWDIR, line_number, 0)
    #     SWDIF = np.delete(SWDIF, line_number, 0)
    #     GLW = np.delete(GLW, line_number, 0)
    CSR = SWDIF / (SWDIF + SWDIR)
    ProcessedDataset[i, :, 0], ProcessedDataset[i, :, 1] = Time, CSR

print(ProcessedDataset.shape)



Coeff = np.asarray([[0.0 for i in range(62)] for j in range(62)])

# Dataset1 = ProcessedDataset[0, :, 1]
# Dataset2 = ProcessedDataset[1, :, 1]
# a = np.corrcoef(Dataset1, Dataset2, rowvar=False)
# coe = a[0,1]
# print(a)
# print(coe - coe % 0.00001)
# print("Dataset1.shape: ", Dataset1.shape)
# print("Datasset2.shape: ", Dataset2.shape)

for j in range(62):
    for k in range(62):
        if j == k:
            continue
        Dataset1 = ProcessedDataset[j, :, 1]
        Dataset2 = ProcessedDataset[k, :, 1]
        # np.correlate returns very big integer around 2400 and I'm not sure the meaning
        Coefficient = np.corrcoef(Dataset1, Dataset2, rowvar=False)
        Coeff[j,k] = Coefficient[0,1]
        

