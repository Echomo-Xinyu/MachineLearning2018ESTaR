'''
Train a model to accurately predict the Clear-sky ratio for anytime during the year. 
The clear-sky ratio is defined as CSR= SWDIF/(SWDIR+SWDIF). So the value runs from 0 to 1. 
When CSR < 0.1, there are no clouds and when CSR> 0.9, it is fully cloudy. 
We will use a bin size of 0.1 for classification. Once you have this model, we will proceed to the next step.
'''
import numpy as np
import pandas as pd

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

Dataset = ReadFile('wrfdata.5')
# print(Dataset)

Time = Dataset[:, 0]
# print(Time)
SWDIR = Dataset[:, 1]
SWDIF = Dataset[:, 2]
GLW = Dataset[:, 3]

# To ignore the possible division by zero
np.seterr(divide='ignore', invalid='ignore')
CSR = SWDIF/(SWDIR+SWDIF)
bin_size = 0.1

# Names for each case is yet to be defined
diff_category = {'case1': 0, 'case2': 0, 'case3': 0, 'case4': 0, 'case5': 0, 'case6': 0, 'case7': 0, 'case8': 0, 'case9': 0, 'case10': 0}
print("The case 1 is for 0-0.1, and case 2 for 0.1-0.2...")
print("The name of each case is yet to be decided")

for element in CSR:
    if element < 0.1:
        diff_category['case1']+=1
    elif element < 0.1+bin_size*1:
        diff_category['case2']+=1
    elif element < 0.1+bin_size*2:
        diff_category['case3']+=1
    elif element < 0.1+bin_size*3:
        diff_category['case4']+=1
    elif element < 0.1+bin_size*4:
        diff_category['case5']+=1
    elif element < 0.1+bin_size*5:
        diff_category['case6']+=1
    elif element < 0.1+bin_size*6:
        diff_category['case7']+=1
    elif element < 0.1+bin_size*7:
        diff_category['case8']+=1
    elif element < 0.1+bin_size*8:
        diff_category['case9']+=1
    elif element < 0.1+bin_size*9:
        diff_category['case10']+=1
    else:
        pass

    
    
for cate_name in diff_category:
    print(cate_name)
    print(diff_category[cate_name])
    print('\n')
