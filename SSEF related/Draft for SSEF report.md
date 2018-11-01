# Draft for SSEF report

### Project Title

Prediction of Solar Power Potential in Singapore using Numerical Weather Prediction Models and Machine Learning

### Abstract

~~Solar energy, being one of the most secure renewable energy, has been widely used in the life, such as agricultural planning, transport, ectricity production and urban plan, etc. Given the high cost of fossil fuel as well as its pollution, the application of solar energy is essential to the development of modern civilisation. Solar techniques include the use of~~

Given the increasing use of solar energy, the prediction of the clean energy is becoming more significant over the time. As there is a colossal amount of data in the past available, Machine Learning Algorithm is playing an important rule in the area. We can derive solar radiation forecat from numerical weather prediction models (NWP) and in this case, clear-sky ratio (CSR), which reflects the ratio relationship between Diffuse SolarRadiation (SWDIF) and Direct Solar Radiation (SWDIR), is employed to measure the solar power potential. A comparison between various machine learning techniques based on historical data in Singapore is performed in this study and in particular we find that the Supervised Vector Machine (SVM) technique is especially useful and perform best on average across two years and among all the four models (to be decided).





Abstract. 

Forecasting solar energy is becoming an important issue in the context of renewable energy sources and Machine Learning Algorithms play an important rule in this field. The prediction of solar energy can be addressed as a time series prediction problem using historical data. Also, solar energy forecasting can be derived from numerical weather prediction models (NWP). Our interest is focused on the latter approach. We focus on the problem of predicting solar energy from NWP computed from GEFS, the Global Ensemble Forecast System, which predicts meteorological variables for points in a grid. In this context, it can be useful to know how prediction accuracy improves depending on the number of grid nodes used as input for the machine learning techniques. However, using the variables from a large number of grid nodes can result in many attributes which might degrade the generalization performance of the learning algorithms. In this paper both issues are studied using data supplied by Kaggle for the State of Oklahoma comparing Support Vector Machines and Gradient Boosted Regression. Also, three different feature selection methods have been tested: Linear Correlation, the ReliefF algorithm and, a new method based on local information analysis.



The increased competitiveness of solar PV panels as a renewable energy source has increased the number of PV panel installations in recent years. In the meantime, higher availability of data and computational power have enabled machine learning algorithms to perform improved predictions. As the need to predict solar PV energy output is essential for many actors in the energy industry, machine learning and time series models can be employed towards this end. In this study, a comparison of different machine learning techniques and time series models is performed across five different sites in Sweden. We find that employing time series models is a complicated procedure due to the non-stationary energy time series. In contrast, machine learning techniques were more straightforward to implement. In particular, we find that the Artificial Neural Networks and Gradient Boosting Regression Trees perform best on average across all sites.


Abstract:

Solar  energy  is  used  in  many  applications,  such  as  increasing  water’s  temperature  or  moving  electrons  in  a  photovoltaic  cell,  agriculture  planning,  fuel  production,  electricity  production,  transport,  architecture  and  urban  planning,  etc.  Solar  energy  is  secure,  clean,  and  available  on  the  Earth  throughout  the  year.  Its  secure  and  clean  applications  are  very  important  to  the  world,  especially  at  a  time  of  fossil  fuel  high  costs  and  the  critical  situation  of  the  atmosphere  resulting  from  fossil  fuel  applications. Solar techniques include the use of phot ovoltaic systems, concentrated solar power and solar water heating to harness the energy. In this paper, prediction is focusing in the Southern part of India and the solar light will be available from 8 to 9 months in a year in this region. So to utilize the solar energy  in an efficient way the prediction is done. To predict the availability of solar energy the machine  learning  Temporal  Gaussian  Process  Regression(TGPR) method  has  been  used.  It  provides  better result and also more robust when compared with the methods like ELM, SVM, etc. The predicted values  are  used  to  measure  and  analyze  the  amount  of  energy  that  could  be  generated  during  a  year  in  the  southern  region  of  India.  This  in  turn  can  be  utilized  to  identify  the  suitable  solar  based  devices  suitable for different locations