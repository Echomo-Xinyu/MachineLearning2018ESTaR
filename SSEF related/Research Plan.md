# Research Plan

## Rationale

With increasing demand of energy supplied to modern society, solar energy, being one of the most important renewable energy, has been put into wide use in the life, such as agricultural planning, electricity production. In order to make best use of solar power and find the place with the strongest solar potential, it is essential to monitor and predict the solar radiation at different places across the country. 

Based on data recorded by the 63 meteorological stations across the country, Numerical Weather Prediction Models (NWP) can be built to predict solar radiation in the future and by finding the correlation between data from different places, the place with estimated strongest solar radiation can hence be found. However, due to the butterfly effect, the NWP models using physical equations can only make very short-term predictions and will be no longer accurate after two weeks' time. 

Given large amount of datasets in the past and current increasing computing power, machine learning algorithms can be applied in this case to make long term solar radiation predictions. By learning features from data in the past, machine learning models can create a general trend over the time and eventually find the place with strongest solar potential by finding the correlation between solar data at various places. There are different machine learning models based on different mechanisms like support vector machine (SVM) and random forest and the focus of this study is to learn about their mechanisms and apply them into this problem.



## Research question(s), hypothesis(es), engineering goal(s), expected outcome

**Research question** is "where is the place across the country with the strongest solar potential?"

**Hypothesis**: Through experiments we can build one or more machine learning models that can make long-term solar potential predictions to make up the absence of Numerical Weather Prediction in long-term prediction.

**Engineering Goal(s)**: To build a few machine learning models that can predict future solar radiation potential and find the place with strongest solar potential (question about the second part)

**Expected outcome**: By building and evaluating several machine learning models, we can find the place with the strongest solar potential.



### Procedures

The rough data is the recordings from 63 meteorological stations across the country. It consists of five columns, Time, Direct Solar Radiation (SWDIR), Diffuse Solar Radiation (SWDIF) and Gradient Level Wind (GLW). Only the first three columns are interested in this study. When both SWDIR and SWDIF are zero, it corresponds to night time.

In order to have a general idea about solar radiation potential in Singapore, graphs of SWDIR and SWDIF values at various locations between different time intervals (a day, a week and a month namely) will be plotted. As the predictions are in terms of CSR, defined as the ratio relationship between SWDIR and SWDIF, the values with both SWDIR and SWDIF zero should be removed. To make the time index more readable compared with number series start from 0, the time index is reformatted to be readable datetime strings. The whole dateset will also be split into training sets and test sets in the ratio of 7:3.

After preprocessing the data, we can start to work on building the model. In order to derive main features of the dataset, including seasonality, spatial correlation and so on and normalize the data, average monthly CSR value will be calculated respectively and substracted from the corresponding data. After this, a few classification models (Support Vector Machine, Random Forest and Time series model) will be used to classify the given time into ten even continuous groups from 0 to 1 based on the given time in the seasonal trend.

The models obtained will be tested based on the test sets seperated ahead and predictions made will be compared against the actual result in the test set. By using statistic figures like mean average error (MAE), mean square error (MSE) and mean absolute percentage error (MAPE), the accuracy of each models can be quantified and compared against one another. The model with the best performance will be chosen to make the future predictions and move to the next stage.

Based on the data in the past at various locations, the correlation coefficient can be calculated with the use of NumPy package. By putting into the predictions made in the previous step, we can create a formula to find the place with the strongest solar potential. (?)



### Risk and Safety

As this project is purely computational, there are very few risks or safety issues involved. 

## Data Analysis

`The following part is only for the model prediction.`

To judge how accurate a model is, usually we set a base line, which make the same guess regardless time or location, and all the models built are compared against it to see how much the model has been improved compared with the random guess. To quantify the improvement, statistical figures such as mean squared error (MSE), mean average error (MAE) and mean absolute percentage error (MAPE). MSE will give higher weight to large variations from the actual result and MAE can be used to measure the overall variation better. MAPE, being one of the most popular figures to use in forecast method accuracy, can be compared with other people's model. Combining the result of three statistics figures, the accuracy of the prediction models can be fully reflected and compared with one another.

## Bibliography

1. Philippe Lauret, Cyril Voyant, Ted Soubdhan, Mathieu David, Philippe Poggi. A benchmarking of machine learning techniques for solar radiation forecasting in an insular context. Solar Energy, Elsevier, 2015, pp.00. <hal-01099211>
2. Seckin Karasu, Aytac Altan, Zehra Sarac, Rifat Hacioglu. Prediction of dolar radiation based on machine learning methods. Bülent Ecevit University (BAP Project No: 2012-17-15-01 and 2014-75737790-01) and International Engineering Research Symposium-UMAS 2017(Duzce University)
3. Cyril Voyant, Gilles Notton, Soteris Ka;pgirou, Marie-Laure Nivet, Cheistophe Paoli, Fabrice Motte, Alexis Fouilloy. Machine Learning methods for solar radiation forecasting: a review. Horizon 2020 project (H2020-LCE-2014-3 - LCE-08/2014 - 646529) TILOS "Technology Innovation for the Local Scale, Optimum Integration of Battery Energy Storage".
4. U. Divya, Chitra Pasupathi. A machine learning approach to predict solar radiation for solar energy based devices. International Journal of Computer Science & Engineering Technology (IJCSET)
5. David J. Lary, Amir H. Alavi, Amir H. Gandhi, Annette L. Walker. Machine learning in geosciences and remote sensing. Geoscience Frontiers 7 (2016) 3-10
6. Sotiris Vardoulakis, Bernard E.A. Fisher, Koulis Pericleous, Norbert Gonzalez-Flesca. Modelling air quality in street canyons: a review. Atmospheric environment, Elsevier, 2003, 37 (2), pp.155-182. <10.1016/S1352-2310(01)00857-9>. <ineris-00961872>
7. Terren-Serrano, Guillermo. "Machine Learning Approach to Forecast Global Solar Radiation Time Series."(2016). http://digitalrepository.unm.edu/ece_etds/249
8. Stephan Rasp and Sebastian Lerch. Neural networks for post-processing ensemble weather forecasts. arXiv: 1805.09091v1 [star.ML] 23 May 2018
9. Jinglin Du, Yayun Liu and Zhijun Liu. Study of Precipitation Forecast Based on Deep Belief Networks. National Nature Science Foundation of China grant number [41575155], and the Prioriyu Academic Program Development of Jiangsu Higher Education Institution(PAPD)
10. Min-Kyu Baek and Duehee Lee. Spatial and Temporal Day-Ahead Tital Daily Solar Irradiation Forecasting: Ensemble Forecasting Based on the Empirical Biasing. 
11. http://fortune.com/2016/09/14/data-machine-learning-solar/ (accessed on 10 Nov 2018)
12. Venugopalan S. G. Raghavan, Harish Gopalan. URB-SolarL An open-source tool for Solar Power Prediction in Urban Areas. 
13. Sanyam Gupta, Infumath K, Govind Singhal. Weather Prediction Using Normal Equation Method and Linear regression Techniques. (IJCSIT) International Journal of Computer Science and Information Technologies, Vol. 7 (3), 2016, 1490-1493. ISSN:0975-9646
14. Numerical Weather Prediction (Weather Models) https://www.weather.gov/media/ajk/brochures/NumericalWeatherPrediction.pdf (accessed on 15 Nov 2018)