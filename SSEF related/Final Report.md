# Final Report

## Introduction

### The significance of solar radiation prediction

<!--This section will talk about the importance of solar radiation power radiation and then link the topic to find an optimal place in Singapore with the strongest solar radiation over the year based on the prediction.-->

Solar radiation data can indicate the strength of the solar power at specific locations on Earth during a specific period of time. As the development of modern society has an increasing high demand of energy and solar energy is one the most important renewable energy sources, the energy supply will increasingly rely on the solar power. The solar radiation data can be of great importance in estimating the solar energy production, thus it is significant to make accurate predictions of solar potential in order to make best use of the energy.

### Available prediction tools

In order to make accurate weather predictions, people have been coming up with all kinds of models. Among them, there are two major categories: physical models known as Numerical Weather Prediction Models (NWP) and machine learning models.

<!--This section will talk about the mechanism of NWP and its limitations, suggesting that machine learning models are used in long-term predictions-->

Currently, the most widely used weather prediction model is known as Numerical Weather Prediction Model (NWP). The NWP uses the computing power to make a forecast of many atmospheric variables such as temperature, pressure, wind and rainfall. The core of the model is the mathematical models based on physical equations with all kinds of real-time data collected by different meteorological stations or weather satellites across the country as the input. By analyzing how the variables will interact with one another based on physical equations, the model will simulate the whole atmospheric environment and make predictions based on different requirements. For example, the NWP model can simulate the movement and thickness of the cloud and based on the real-time simulation, it can predict the solar potential of the region with very high accuracy because the environment simulated is very close to the actual world and they are based on the same physical principles.

The NWP model is commonly used to make short-term weather predictions and long-term climate predictions. One of the reasons that restrict the NWP’s prediction time period is that the model requires very huge computing powers. Manipulating vast amounts of data and performing complex calculation requires some of the most powerful supercomputers in the world. Even with the increasing computer power, the forecast by the model can only remain accurate for around 6 days. Moreover, the more fundamental reason lies under the chaotic natural environment. It is impossible to solve all the partial differential equations of the atmosphere exactly and small errors during each step will gradually influence the accuracy of the model. As a result, the NWP model will no longer be accurate after 14 days. Despite its low accuracy in long-term weather prediction, the NWP model requires huge amount of prior knowledge to interpret the computer forecast and because of this, very few people can operate the model to make the predictions as they want.

That is where the machine learning comes in. With huge amount of datasets available, people realise that computers can be fed with the data and find the general trend of the data. Instead of simulating a real-time atmospheric environment, machines can extract features from the dataset and find the relationships lying under the figures. In this way, machines can build various models based on different algorithms and make predictions of the solar radiation at any given time and location. The focus of this study will be how to build a model with high performance to make prediction of future solar radiation with last year’s data collected from 62 meteorological stations across Singapore.

## Methodology

### Data pre-processing

<!--Removing the zero value as it would influence the fit of the model. Convert the given index number to the human readable datetime floats Manipulate the given index into the input required, specially for the time-series model as it requires datetime object input. Data normalizing to zero mean and 1 standard deviation-->

The initial overall dataset consists of solar-related data collected from 62 different meteorological stations across Singapore for one past year. There are four columns in each dataset: Time, Direct Solar Radiation (SWDIR), Diffuse Solar Radiation (SWDIF) and Gradient Level Wind (GLW) and only first three columns are interested in this study. When both SWDIR and SWDIF are zero, it corresponds to the night time when there is no solar radiation at the location. 

In this study, the solar potential is measured by clear-sky ratio (CSR), defined as $CSR = SWDIR / (SWDIR+SWDIF)$. As Singapore has very limited land area, if there is no cloud cover, the total amount of radiation from the sun is rather constant across the country and hence CSR can be an applicable measure of solar potential across the country. In order to represent the solar radiation intuitively, the CSR value has been divided into ten continuous even groups (Group 1 to Group 10) from 0 to 1, namely $0.0-0.1, 0.1-0.2, 0.2-0.3 … 0.9 - 1.0$.

The CSR values at night, which are expected to be NaN (not a number) by its definition, are classified to Group 0, in order to maintain the continuity of the dataset. Otherwise, the processed data is no longer continuous. As the length of the night time may vary slightly from day to day, the model building can be influenced due to empty data present.

In order to reduce the requirement of users, the Time indexes, given in the format of floats, have been converted to readable string-like floats. For example, the original 2nd time index is *0.254167*, which can be hard to understand as no one really knows how it measures the time, has been converted to "*201801010815*", which can be understood by people. Due to the requirement of the Sci-kit package used in the study, the inputs are supposed to be in the format of floats, so the format is maintained in the processed dataset, but in a human-readable way.

For certain models, time series model in this case, the model requires specific datetime-type object input, By calculating the time interval of the data collection, a series of datetime-type object th known start, end and frequency is generated to fulfill the requirement of the model.

The whole dataset has been normalized in the mean and standard deviation. X<sub>norm(i,j)</sub> = (X<sub>(i,j)</sub> - mean(X<sub>j</sub>)) / std(X<sub>j</sub>), where X<sub>j</sub> is the j<sup>th</sup> column of matrix X. After normalising the dataset, the mean becomes zero and has unit variance along each feature and can be less sensitive to the scale of features.



## Various Models Used

###Classification models

#### Support Vector Machine

#### Random Forest Regressor

####Artificial Neural Network

###Regression Models

#### Time-series model

####Random Forest Regressor

####K-Neatest neighbour

## Evaluation and Application of results

#### Evaluation

Assessment of the accuracy of different models in terms of different statistical figures like MAE, MSE, RMSE, MAPE.

The sequence is planned to be: 1. introduce different assessment tools like MAPE score, average accuracy of 10*10 cross validation. 2. Compare scores of different models in the format of a table.

## Conclusion and outlook

I think there is not much to plan for the conclusion. For the future work part, I have prepared a few essays (while finding the similar essays) as the future work. The range of future work includes: try deep belief network, comparison regarding more aspects not just restricted accuracy but running time etc, (neural network model)