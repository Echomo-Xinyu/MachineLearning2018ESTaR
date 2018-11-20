# Report Skeleton

## Introduction

### The significance of solar radiation prediction

With increasing demand of energy across the world, the energy supply will increasingly rely on the renewable energy sources like solar energy, wind energy and so on. 

`This section will talk about the improtance of solar radiation power radiatin and then link the topic to find an optimal place in Singapore with the strongest solar radiation over the year based on the prediction.`

### Available prediction tools

The tools to predict solar radiation are mainly under two categories, physical models known as numerical weather prediction (NWP) and machine learning models.

`This section will talk about the mechanism of NWP and its limitations, suggesting that machine learning models are used in long-term predictions.`

## Methodology

### Data pre-processing

`Removing the zero value as it would influence the fit of the model. `

`Convert the given index number to the human readable datetime floats`

`Manipulate the given index into the input required, specially for the time-series model as it requires datetime object input.`

`Data normalizing to zero mean and 1 standard deviation`

### Various models used

`For the following models talked about, the explaining sequence is going to be its mechanism (with figures to explain if necessary), and explain its common advantages and disadvantages as well as its limitations in this case (mainly for time-series model as it works for longer time interval)`

#### Support Vector Machine



#### Random Forest



#### Time-series model



### Regularization

`I only add in this section because I find some essays include this part and I find it would be quite useful to talk about. Here this section will talk about the importance of regularization, introducing the concept of over-fit and under-fit. And maybe use one model svm as example to illustrate the importance of regularization.`





## Evaluation and Application of results

#### Evaluation

`Assessment of the accuracy of different models in terms of different statistical figures like MAPE.`

`The sequence is planned to be: 1. introduce different assessment tools like MAPE score, average accuracy of 10*10 cross validation. 2. Compare scores of different models in the format of a table.`



#### Application

`Correlate the prediction across the year at different locations and reach the conclusion to find a place with preidcted strongest solar potential power.`



## Conclusion and outlook

`I think there is not much to plan for the conclusion. For the future work part, I have prepared a few essays (while finding the similar essays) as the future work. The range of future work includes: try deep belief network, comaprison regarding more aspects not just restricted accuracy but running time etc, (neural network model)`