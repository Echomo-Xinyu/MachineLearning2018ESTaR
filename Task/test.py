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
index = 1
TotalDataset = np.asarray([[0 for i in range(4)] for j in range(35040)])
ProcessedDataset = np.asarray([[0 for i in range(2)] for j in range(35040)])
pathway = "/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.1"
TotalDataset = np.asarray(ReadFile(pathway))
Time = TotalDataset[:, 0]
SWDIR = TotalDataset[:, 1]
SWDIF = TotalDataset[:, 2]
GLW = TotalDataset[:, 3]

lis = list()
# @a @b represent the start and end of time interval
a, b = 0, 0
# @index the index number of 96
index = 0
indexa, indexb = 0, 0
currentseasonindexa, currentseasonindexb = 0, 0
# used to know how many days for this periof
daycounter = 0
overalldaycounter = 0
n = np.size(Time, 0)
for i in range(n):
    if SWDIR[i] == 0.0 and SWDIF[i] == 0.0:
        index = i % 96
        if a == 0 and b == 0:
            a, b = i, i
            indexa = index
        elif i == b + 1:
            b += 1
            indexb = index
        else:
            daycounter += 1
            overalldaycounter += 1
            if currentseasonindexa == 0 and currentseasonindexb == 0:
                currentseasonindexa, currentseasonindexb = a, b
            else:
                if currentseasonindexa == indexa and currentseasonindexb == b:
                    pass
                elif currentseasonindexa != indexa or currentseasonindexb != indexb:
                    print("From this moment, the current season index a and b have been changed.")
                    daycounter -= 1
                    lis.append((daycounter, currentseasonindexa, currentseasonindexb, overalldaycounter))
                    currentseasonindexa = indexa
                    currentseasonindexb = indexb
                    print("The new season index a is: ", indexa)
                    print("The new season index b us: ", indexb)
                    daycounter = 1
            print("This is the end of this time interval(day): ")
            print(a)
            print(b)
            # print("Time start and end is: ", a, " " ,b)
            # print("The corresponding indexes are: ", indexa, " ", indexb)
            print(indexa)
            print(indexb)
            a, b = 0, 0
            indexa, indexb = 0, 0

for li in lis:
    print(li)



# print("This is the end.")
#         count_line.append(i)
# count_line.sort()
# for number in reversed(range(count)):
#     line_number = count_line[number]
#     TimeIndex = Time[line_number]
#     a = TimeIndex % 96
#     if a < 46 or a > 94:
#         print("Strange input")
#         print("The input is: ", number)
# CSR = SWDIF / (SWDIF + SWDIR)
# ProcessedDataset[:, 0], ProcessedDataset[:, 1] = Time, CSR