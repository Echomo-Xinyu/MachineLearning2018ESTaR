# Random Notes for report

* https://www.ema.gov.sg/solar_photovoltaic_systems.aspx

  > Solar energy remains the most promising renewable energy source for Singapore when it comes to electricity generation. With an average annual solar irradiance of 1,580 kWh/m2/year and about 50 percent more solar radiation than temperate countries, solar photovoltaic (PV) generation has the greatest potential for wider deployment in Singapore.

* 

Random Forests are trained via the bagging method. Bagging or Bootstrap Aggregating, consists of randomly sampling subsets of the training data, fitting a model to these smaller data sets, and aggregating the predictions. This method allows several instances to be used repeatedly for the training stage given that we are sampling with replacement. Tree bagging consists of sampling subsets of the training set, fitting a regression Decision Tree to each, and aggregating their result.

The Random Forest method introduces more randomness and diversity by applying the bagging method to the feature space. That is, instead of searching greedily for the best predictors to create branches, it randomly samples elements of the predictor space, thus adding more diversity and reducing the variance of the trees at the cost of equal or higher bias. The process is known as “feature bagging” and can help lead to a more robust model.

Known that in a Decision Tree a new instance goes from the root node to the bottom until it is classified in a leaf node. In the Random Forests algorithm, each new data point goes through the same process, but now it visits all the different trees in the ensemble, which are were grown using random samples of both training data and features. Depending on the task at hand, the functions used for aggregation will differ. For Classification problems, it uses the mode or most frequent class predicted by the individual trees (also known as a majority vote), whereas for Regression tasks, it uses the average prediction of each tree.



**Difference 1: When to use classification vs regression tree**

This might seem like a trivial issue - once you know the difference! **Classification** **trees**, as the name implies are used to separate the dataset into classes belonging to the response variable. Usually the response variable has two classes: Yes or No (1 or 0). If the target variable has **more** than 2 categories, then a variant of the algorithm, called C4.5, is used. For binary splits however, the standard CART procedure is used. Thus classification trees are used when the response or target variable is categorical in nature. 

**Regression** **trees** are needed when the response variable is numeric or continuous. For example, the predicted price of a consumer good. Thus regression trees are applicable for *prediction* type of problems as opposed to *classification*.

Keep in mind that in either case, the predictors or independent variables may be categorical or numeric. It is the **target variable** that determines the type of decision tree needed.

**Difference 2: How they work**

In a standard classification tree, the idea is to split the dataset based on homogeneity of data. Lets say for example we have two variables: age and weight that predict if a person is going to sign up for a gym membership or not. In our training data if it showed that 90% of the people who are older than 40 signed up, we split the data here and age becomes a top node in the tree. We can almost say that this split has made the data "90% pure". Rigorous measures of impurity, based on computing proportion of the data that belong to a class, such as entropy or Gini index are used to quantify the homogeneity in Classification trees.

In a regression tree the idea is this: since the target variable does not have classes, we fit a regression model to the target variable using each of the independent variables. Then for each independent variable, the data is split at several split points. At each split point, the "error" between the predicted value and the actual values is squared to get a "Sum of Squared Errors (SSE)". The split point errors across the variables are compared and the variable/point yielding the lowest SSE is chosen as the root node/split point. This process is recursively continued.



**Multi-layer Perceptron (MLP)** is a supervised learning algorithm that learns a function f(⋅):Rm→Ro by training on a dataset, where m is the number of dimensions for input and o is the number of dimensions for output. Given a set of features X=x1,x2,...,xm and a target y, it can learn a non-linear function approximator for either classification or regression. It is different from logistic regression, in that between the input and the output layer, there can be one or more non-linear layers, called hidden layers. Figure 1 shows a one hidden layer MLP with scalar output.

The leftmost layer, known as the input layer, consists of a set of neurons {xi|x1,x2,...,xm} representing the input features. Each neuron in the hidden layer transforms the values from the previous layer with a weighted linear summation w1x1+w2x2+...+wmxm, followed by a non-linear activation function g(⋅):R→R - like the hyperbolic tan function. The output layer receives the values from the last hidden layer and transforms them into output values.

The module contains the public attributes `coefs_` and `intercepts_`. `coefs_` is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1. `intercepts_` is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.

MLP trains using [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), [Adam](https://arxiv.org/abs/1412.6980), or [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS). Stochastic Gradient Descent (SGD) updates parameters using the gradient of the loss function with respect to a parameter that needs adaptation, i.e.

$w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w} + \frac{\partial Loss}{\partial w})$

where η is the learning rate which controls the step-size in the parameter space search. Loss is the loss function used for the network.

More details can be found in the documentation of [SGD](http://scikit-learn.org/stable/modules/sgd.html)

Adam is similar to SGD in a sense that it is a stochastic optimizer, but it can automatically adjust the amount to update parameters based on adaptive estimates of lower-order moments.

With SGD or Adam, training supports online and mini-batch learning.

L-BFGS is a solver that approximates the Hessian matrix which represents the second-order partial derivative of a function. Further it approximates the inverse of the Hessian matrix to perform parameter updates. The implementation uses the Scipy version of [L-BFGS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html).

If the selected solver is ‘L-BFGS’, training does not support online nor mini-batch learning.



### Back-Propagation

Allows the information to go back from the cost backward through the network in order to compute the gradient. Therefore, loop over the nodes starting at the final node in reverse topological order to compute the derivative of the final node output with respect to each edge’s node tail. Doing so will help us know who is responsible for the most error and change the parameters in that direction. The following derivatives’ formulas will help us write the back-propagate functions: Since *b^l* is always a vector, the sum would be across rows (since each column is an example).



The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

![img](https://cdn-images-1.medium.com/max/1200/0*9jEWNXTAao7phK-5.png) ![img](https://cdn-images-1.medium.com/max/1200/0*0o8xIA4k3gXUDCFU.png) 

To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

#### Hyperplanes and Support Vectors

![img](https://cdn-images-1.medium.com/max/1600/1*ZpkLQf2FNfzfH4HXeMw4MQ.png)

Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. It becomes difficult to imagine when the number of features exceeds 3.

Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.

#### Cost Function and Gradient Updates

In the SVM algorithm, we are looking to maximize the margin between the data points and the hyperplane. The loss function that helps maximize the margin is hinge loss.![img](https://cdn-images-1.medium.com/max/1200/1*3xErahGeTFnbDiRuNXjAuA.png) 

![img](https://cdn-images-1.medium.com/max/1200/1*hHlytjVk6d7O2WWvG2Gdig.png) 

The cost is 0 if the predicted value and the actual value are of the same sign. If they are not, we then calculate the loss value. We also add a regularization parameter the cost function. The objective of the regularization parameter is to balance the margin maximization and loss. After adding the regularization parameter, the cost functions looks as below.![img](https://cdn-images-1.medium.com/max/1600/1*GQAd28bK8LKOL2kOOFY-tg.png) Now that we have the loss function, we take partial derivatives with respect to the weights to find the gradients. Using the gradients, we can update our weights.![img](https://cdn-images-1.medium.com/max/1600/1*WUphtYLfTOAoaXQXvImBeA.png) When there is no misclassification, i.e our model correctly predicts the class of our data point, we only have to update the gradient from the regularization parameter.![img](https://cdn-images-1.medium.com/max/1600/1*-nKEXrWos8Iuf-DWSv_srQ.png) 

When there is a misclassification, i.e our model make a mistake on the prediction of the class of our data point, we include the loss along with the regularization parameter to perform gradient update.

![img](https://cdn-images-1.medium.com/max/1600/1*tnvMhAKaTUCO43diEvtTAQ.png) 



The basic nearest neighbors regression uses uniform weights: that is, each point in the local neighborhood contributes uniformly to the classification of a query point. Under some circumstances, it can be advantageous to weight points such that nearby points contribute more to the regression than faraway points. This can be accomplished through the `weights`keyword. The default value, `weights = 'uniform'`, assigns equal weights to all points. `weights = 'distance'` assigns weights proportional to the inverse of the distance from the query point. Alternatively, a user-defined function of the distance can be supplied, which will be used to compute the weights.![sphx_glr_plot_regression_0011](../../../Desktop/sphx_glr_plot_regression_0011.png)

[^3][^1][^2][^6][^15][^13] [^8][^9]

### Bibliography

[^4 27]: U. Divya, Chitra Pasupathi. A machine learning approach to predict solar radiation for solar energy based devices. By International Journal of Computer Science & Engineering Technology (IJCSET) 
[^3 28]: Cyril Voyant, Gilles Notton, Soteris Kalogirou, Marie-Laure Nivet, Christophe Paoli, Fabrice Motte, Alexis Fouilloy. Machine Learning methods for solar radiation forecasting: a review. Horizon 2020 project (H2020-LCE-2014-3 - LCE-08/2014 - 646529) TILOS "Technology Innovation for the Local Scale, Optimum Integration of Battery Energy Storage".
[^18 20]: T.M.Giannaros, V.Kotroni, and K.Lagouvardos. Predicting lightning activity in Greece with the weather research and forecasting (wrf) model. Atmospheric Research, vol. 156. 1 - 13, 2015.
[^7 21]: Terren-Serrano, Guillermo. "Machine Learning Approach to Forecast Global Solar Radiation Time Series."(2016).[ http://digitalrepository.unm.edu/ece_etds/249] (Stephan Rasp and Sebastian Lerch. Neural networks for post-processing ensemble weather forecasts. arXiv: 1805.09091v1 [star.ML] 23 May 2018)
[^19 29]: D.Carvalho, A. Rocha, M. Gomez-Gesterira, and C. S. Santos. Sensitivity of the WRD model wind simulation and wind energy production estimates to planetary boundary layer parameterizations for onshore and offshore areas in the Iberian peninsula. Applied Energy, vol. 135, pp. 234-246, 2014. 
[^20 30]: A.M. Guerrero-Higueras, E.Garcia-Ortega, J.Lorenzana and V.Matellan. Schedule WRF model executions in parallel computing environments using Python.
[^1 31]: Philippe Lauret, Cyril Voyant, Ted Soubdhan, Mathieu David, Philippe Poggi. A benchmarking of machine learning techniques for solar radiation forecasting in an insular context. Solar Energy, Elsevier, 2015, pp.00. <hal-01099211>
[^2 32]: Seckin Karasu, Aytac Altan, Zehra Sarac, Rifat Hacioglu. Prediction of solar radiation based on machine learning methods. Bülent Ecevit University (BAP Project No: 2012-17-15-01 and 2014-75737790-01) and International Engineering Research Symposium-UMAS 2017(Duzce University)
[^6 33]: Sotiris Vardoulakis, Bernard E.A. Fisher, Koulis Pericleous, Norbert Gonzalez-Flesca. Modelling air quality in street canyons: a review. Atmospheric Environment, Elsevier, 2003, 37 (2), pp.155-182.
[^8 34]: Stephan Rasp and Sebastian Lerch. Neural networks for post-processing ensemble weather forecasts. arXiv: 1805.09091v1 [star.ML] 23 May 2018
[^9 35]: Min-Kyu Baek and Duehee Lee. Spatial and Temporal Day-Ahead Total Daily Solar Irradiation Forecasting: Ensemble Forecasting Based on the Empirical Biasing
[^12 36]: Venugopalan S.G. Raghavan, Harish Gopalan. URB-Solar An open-source tool for Solar Power Prediction in Urban Areas
[^13 37]: Sanyam Gupta, Infumath K, Govind Singhal. Weather Prediction Using Normal Equation Method and Linear regression Techniques. (IJCSIT)
[^15 38]: Wei-Zhen Lu, Wen-Jian Wang. Potential assessment of the “Support Vector Machine” method in forecasting ambient air pollutant trend
[^16 39]: Xie-Kang Wang, Wei-Zhen Lu. Seasonal Variant of air pollution index: Hong Kong case study
[^17 40]: Ricardo Aler, Ricardo Martin, Jose M. Valls, and Ines M. Galvan. A Study of Machine Learning Techniques for Daily Solar Energy Forecasting using Numerical Weather Models.
[^11 41]: <http://fortune.com/2016/09/14/data-machine-learning-solar/> (accessed on 10 Nov 2018)
[^14 42]: Numerical Weather Prediction (Weather Models)[ https://www.weather.gov/media/ajk/brochures/NumericalWeatherPrediction.pdf](https://www.weather.gov/media/ajk/brochures/NumericalWeatherPrediction.pdf) (accessed on 15 Nov 2018)
[^21 43]: Tolstykh, Mikhail & Frolov, Alexander. (2005). Some Current Problems in Numerical Weather Prediction. Izvestiya Atmospheric and Oceanic Physics. 41. 285-295.  
[^22 22]: Haupt, Sue & Kosovic, Branko. (2015). Big Data and Machine Learning for Applied Weather Forecasts Forecasting Solar Power for Utility Operations. 10.1109/SSCI.2015.79.  
[^23 23]: Aoife M.Foleyabd, Paul G.Leahyab, AntoninoMarvugliac, Eamon J.McKeoghabCurrent methods and advances in forecasting of wind power generation 
[^24 24]: https://www.ema.gov.sg/solar_photovoltaic_systems.aspx (accessed on 20 Nov 2018) 
[^25 25]: Ran Fu, David Feldman, Robert Margolis, Mike Woodhouse, and Kristen Ardani. U.S. solar photovoltaic system cost benchmark: Q1 2017. Sep.



### Regression Models

Regression process is usually used to estimate the relationship between a dependent variable and one or more independent variables. More specifically, it can help people understand how the dependent variable changes when any one of the independent variables is varied, while the other independent variables remain fixed.[^2] In this study,  two regression models are used and the section will explain the algorithms behind each model.



From the table above, we can clearly see that the SARIMAX and KNN models have significantly higher performance than SVC and RFC models with the corresponding error metrics below half of the latter. This could be because the regression model makes use of the continuity of the training dataset. Compared classifying the solar radiation data into different groups based on the given time index and other variables, the regression model can better reflect the relationship between features $X$ and the solar radiation data $y$ and thus can make much more accurate predictions than the classification. The poor performance of the classification models can also be caused by inappropriate selection of features and this area can be further explore。

In between two regression models, the SARIMAX model has very similar performance as the KNN model, around 1% lower in MAE and around 2% higher in RMSE. This could be the result of making use of the seasonal trend. The solar radiation data can be influenced by the change of the relative locations between the sun and the earth as well as other factors. Realising the seasonal trend and making use of the autocorrelation and expectation of the time series can contribute to reducing the overall difference between prediction and true values, thus reducing the MAE value. However, if we zoom in to all the individual predictions, a large difference can exist between some predictions and corresponding actual radiation data and hence leads to higher RMSE of the SARIMAX than the KNN.



30,177; 540, 415