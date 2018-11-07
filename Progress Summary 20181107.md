# Progress Summary 20181107

### Latest Version of Abstract

Given the increasing use of solar energy, the prediction of the clean energy is becoming more significant over the time. We can derive solar radiation forecast from numerical weather prediction models (NWP) and in this case, clear-sky ratio (CSR), which reflects the ratio relationship between Diffuse SolarRadiation (SWDIF) and Direct Solar Radiation (SWDIR), is employed to measure the solar power potential. However, despite its high accuracy, NWP, considered to be computer power consuming, can only predict future solar radiation in six days and would be no longer accurate after 14 days. In order to get the long-term results quick yet similarly accurate, Machine Learning Algorithms, which take up far less computing power and can obtain long-term general trends, is now playing an important role in the area. In this paper, a few machine learning techniques like Support Vector Machine(SVM), Random Forest and Artificial Neural Network(ANN) together with the time series modes are going to be compared regarding their performance in prediction. Moreover, various feature selection including Linear Correlation and ReliefFalgorithm are used in this study.

### Update progress with the data given (from last summary onwards)

###### what I have done

* Convert the given data series (1, 2, .. 105120) to the datetime formate (2015-01-01 08:00, .. 2016-01-01 7:55) also called time series data
* Apply Dataframe from pandas to the data process
* Test the stationary of the time series data, it seems very stationary and thus require no further processing like log (though I try the log to make the data smoother)
* Fit the time series model with the time series data but FAILed as the time series data are not continuous in its frequency, the strange CSR values (NaN) are removed during the data pre-processing as it would stop the fit of model to the data

###### Plan to do later

As I have read the paper, I'm thinking about <u>building one or two more models</u> like *Gradient Boost Regression and artificial neural network* and then go to the comparison step or <u>apply all the models to the new big dataset</u> provided and <u>work on the correlation</u> between different places.



### Schedules for visits to IHPC

As this is already Wednesday..

Tomorrow (Thursday) I'm going to have my higher mother tongue exam till 1pm and after that I think there's a make up session for the NOI which was planned on Tuesday but cancelled for Deepvali.