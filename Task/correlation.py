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

TotalDataset = np.asarray([[[0 for i in range(4)] for j in range(35040)] for k in range(63)])
for i in range(62):
    index = str(i+1)
    pathway = "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata." + index
    TotalDataset[i] = np.asarray(ReadFile(pathway))
    Time = TotalDataset[i, :, 0]
    SWDIR = TotalDataset[i, :, 1]
    SWDIF = TotalDataset[i, :, 2]
    GLW = TotalDataset[i, :, 3]
    count = 0
    count_line = list()
    n = np.size(Time, 0)
    for i in range(n):
        if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
            count+=1
            count_line.append(i)
    count_line.sort()
    for number in reversed(range(count)):
        line_number = count_line[number]
        Time = np.delete(Time, line_number, 0)
        SWDIR = np.delete(SWDIR, line_number, 0)
        SWDIF = np.delete(SWDIF, line_number, 0)
        GLW = np.delete(GLW, line_number, 0)

print(TotalDataset.shape)



# for j in range(63):
#     for k in range(63):
#         if j == k:
#             continue
#         Dataset1 = TotalDataset[j]
#         Dataset2 = TotalDataset[k]


