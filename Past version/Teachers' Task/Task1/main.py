import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Initialize the four variables we're gonna use
Time = Dataset[:, 0]
# print(Time)
SWDIR = Dataset[:, 1]
SWDIF = Dataset[:, 2]
GLW = Dataset[:, 3]

# To ignore the possible division by zero
np.seterr(divide='ignore', invalid='ignore')
CSR = SWDIF/(SWDIR+SWDIF)
print(CSR)

bin_size = 0.1

# def estimateGaussian(X):
#     n = np.size(X, 0)
#     # print(n)
#     mean = np.zeros((n, 1))
#     sigma2 = np.zeros((n, 1))

#     mu = mean(X).transpose()
#     sigma2 = np.cov(X, 1)
#     return mu, sigma2


# mu, sigma2 = estimateGaussian(CSR)

n = np.size(CSR, 0)
mu = np.zeros((n, 1))
sigma2 = np.zeros((n, 1))
Time = Time.transpose()

X = np.vstack((Time, CSR))
# X is 2D matrix with first row for time and second row for CSR

mu = np.mean(X, axis=0) # the current mu is a vector with length of 105120

print("original sigma2:")
print(sigma2.shape)
sigma2 = np.cov(X, ddof=1)
print("Current sigma2: ")
print(sigma2.shape)
print(sigma2)
# sigma2 = np.std(X, ddof=1)


# print("sigma2: ")
# print(sigma2)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))