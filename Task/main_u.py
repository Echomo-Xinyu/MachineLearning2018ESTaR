'''
This script is under the instructions from Dr Harish. The instructions are below:
A single model will not work for the entire dataset.
If we had clear skies daily, the Gaussian model will be a good fit to the whole data collection.
You need to apply the model for every 24 hours. (leave out the first 16 hours).
If the correlation is greater than ~ 0.75, you can assume that the fit is good.
However, for days in which there are cloudy skies (common in Singapore),
you need to detect the sudden drop in the radiation and factor it approximately into the model.
This will minimize the error of your predictions.
The skeleton for model construction reads as follows:

Direct = f (time, cloudiness)
Diffuse = g(time,cloudiness)

where cloudiness = Diffuse / (Direct + Diffuse) from the data given to you.
When cloudiness is less than 0.2, you can use a Gaussian fit for both direct and diffuse.
Beyond this cloudiness, you need to think of a way for constructing the model.
You need to start exploring the different model construction options 
and start thinking of ways to represent the data.
For example, you can look into how Perez Model reconstructs solar radiation.
Also the common tools for solar radiation prediction are the hidden Markov model and support vector regression.
You need to think of a way of using the data with these models 
and let us know if you need additional inputs.
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
# import seaborn as sns
import datetime

from matplotlib import rcParams
#图片大小
rcParams['figure.figsize'] = (10, 6)
#图片精度
rcParams['figure.dpi'] = 150
#线宽
rcParams['lines.linewidth'] = 2
#是否显示网格
rcParams['axes.grid'] = True
#axes 填充颜色为浅灰
rcParams['axes.facecolor'] = '#eeeeee'
#字体大小为14
rcParams['font.size'] = 14
#边框为无色
rcParams['patch.edgecolor'] = 'none'

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
RoughDataset = ReadFile('/Users/ue/Downloads/MachineLearning2018ESTaR/Dataset/wrfdata.5')

FirstDayData = RoughDataset[16:40,:]
print("Data for the first day:\n",FirstDayData)

Time = FirstDayData[:,0]
SWDIR = FirstDayData[:, 1]
SWDIF = FirstDayData[:, 2]
GLW = FirstDayData[:, 3]

CSR = SWDIF / (SWDIF + SWDIR)
print("CSR:\n", CSR)

dataSize = np.size(Time, 0)
print("total number of useful datas is: ", dataSize)

def gaussian(x, a, mu, sig, offset=0):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + offset

def plotGraph(x_para, y_para):
    print("Now the computer is drawing the graph. This may take a bit long time. Please wait patiently and take your dinner if you haven't.:)")
    # popt, pcov = curve_fit(gaussian, x_para, y_para, p0=[np.max(y_para), np.median(x_para), np.std(x_para), np.min(y_para)])
    popt, pcov = curve_fit(gaussian, x_para, y_para)
    # plot original data
    plt.plot(x_para, y_para, 'b,', label='data')
    # plot fit function
    x_fit = np.linspace(np.min(x_para), np.max(x_para), 1000)
    plt.plot(x_fit, gaussian(x_fit, *popt), 'r-', label='fit')
    plt.legend()
    plt.title('Fig for CSR against Time on the first day')
    plt.xlabel('Time')
    plt.ylabel('CSR')

    end_time = datetime.datetime.now()
    print("Time taken to run the program till complete the graph: ", (end_time-start_time).seconds, " seconds")

    plt.show()

# plotGraph(Time, CSR)
# plt.plot(Time, CSR, 'b,', label='data')
# plt.show()

def scatterplot(x_data, y_data, x_label="Time", y_label="CSR", title="Figure for the first day", color = "r", yscale_log=False):


    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    plt.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

    # Label the axes and provide a title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

scatterplot(Time, CSR)