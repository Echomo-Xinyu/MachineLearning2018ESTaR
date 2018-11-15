import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
# import seaborn as sns
import datetime
from sklearn import svm


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

max = 0.00000

for element in RoughDataset[:,1]:
    if element > max:
        max = element

print("The largest value in the second column is: ", max)

max = 0.00000
for element in RoughDataset[:,2]:
    if element > max:
        max = element

print("The largest value in the third column is: ", max)

max = 0.00000
for element in RoughDataset[:,3]:
    if element > max:
        max = element

print("The largest value in the fourth column is: ", max)
