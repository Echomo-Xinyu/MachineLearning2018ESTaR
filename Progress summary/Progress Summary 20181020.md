#  Progress Summary 20181020

### Update on progress with the data given

###### Description of the data

The data given includes four columns of information: Time, SWDIR (Direct Solar Radiation) SWDIF (Diffuse SolarRadiation) and GLW (not used so far) in the format of 4 * 105120 . It records the SWDIR and SWDIF from Jan 1 2018 8 AM to Jan 1 2019 8 AM. When both SWDIR and SWDIF are zero, it corresponds to the night time. The clear-sky ratio is defined as CSR = SWDIF / (SWDIF + SWDIR) and the task is to make a prediction of the CSR for anytime during the year.

###### What I have done to the data: 

So far I have moved all the daytime data (useful data) into a new set of data for classification. 

1. I first try to **apply the gaussian model** to make the prediction. But what I eventually get using this method is a straight line parallel to the x-axis and the value is just in the middle of all the extreme points and cannot really make the predictions accurately. ![Figure_2](/Users/SunXinyuSingapore/Downloads/MachineLearning2018ESTaR/TeachersTask/Task1/Figure_2.png)
2. I have **plotted all the useful data** in a figure with y-axis representing the CSR and x-axis representing the time to study their distribution. As shown above, we can see the data generally concentrated between 0.1 and 0.2 but the distribution in the upper part of the figure is also very uniform, which suggests that I cannot simply take them in the modelling. Then I turn to find the trend for a smaller range of time and make the comparison between models for example each month.
3. Then I restrict my data to a smaller range and **only plot the data from the first month**. I normalise the data before plotting so that the time data looks simpler without starting from 3000. I also tried to **fit the gaussian model** in this case but find it very hard or even ridiculous as they are not concentrated in certain parts (Correct me if I have any misunderstanding regarding use of gaussian model) and that's where i have stopped. ![Month1Data](/Users/SunXinyuSingapore/Downloads/MachineLearning2018ESTaR/TeachersTask/Task1/Month1Data.png)

######Plan to do

1. As suggested in the point 3 in the last part and after reading some papers, I'm considering using a **hybrid model combing gaussian model and other methods**. (I cannot find the paper I have read before the exam now... I'll quickly go through during the weekend and find it cough cough) The reason to choose the hybrid model with gaussian model is due to its ease to use and very low requirement of computer power or time as well as its extensive use. Yet the problem is that it simplifies the complex problem and it may be hard to take in consideration of complex environments like the influence of the wind speed.

2. After plotting the data, I just recall the lesson on Coursera to use **supervised learning** method like **support vector machine or polynominal regression** method to draw the best fit curve.  There are four main **advantages**: Firstly it has a regularisation parameter, which makes the user think about avoiding over-fitting. Secondly it uses the kernel trick, so you can build in expert knowledge about the problem via engineering the kernel. Thirdly an SVM is defined by a convex optimisation problem (no local minima) for which there are efficient methods (e.g. SMO). Lastly, it is an approximation to a bound on the test error rate, and there is a substantial body of theory behind it which suggests it should be a good idea. The **disadvantages** are that the theory only really covers the determination of the parameters for a given value of the regularisation and kernel parameters and choice of kernel. In a way the SVM moves the problem of over-fitting from optimising the parameters to model selection. Sadly kernel models can be quite sensitive to over-fitting the model selection criterion.

3. I've also seeked advice from my friends and get the suggestion of **random forest**, but I haven't learned enough yet, just read it in an essay (and I also cannot find the essay right now hmmm) so I shall just put a screenshot of one answer I find on Quora: 



   ![image-20181019190407809](/Users/SunXinyuSingapore/Library/Application Support/typora-user-images/image-20181019190407809.png)


That should probably be all I have got before my exams and I'll try to find the two useful essays mentioned and build the hybrid model and the SVM one during the coming weekend.

### Description of the general problem statement

The purpose of the experiments above is to make predictions regarding furture clear-sky ratio anytime based on past data we have collected. (And this is only the initial task) 

With the bin size of 0.1, I'm going to seperate the useful dataset into three parts, training set, cross-validation set and test set. Training set is to train the model and cross validation set is to make comparisons between different models I have built and chose the best one in order to avoid the problem of overfitting. Then use the test set to decide the accuracy of the model. This is the current planned evaluation method I have.

With the cross validation, we make sure the hypothesis is not overfitting to the training set of the data and thus the hypothesis is more likely to describe the general trend and is capable to make predictions for the future years rather than only this specific year.

### Schedule for visits to IHPC

For the upcoming week, I'm going to have MTL intensive camp and the ending day for every day is 1:30 PM. So I shall come to IHPC every day after my school and probably start by 1:45 if I can.

**Monday: **2:15pm - 7pm

**Tuesday: **Not coming for NOI session from 16:30-18:30

**Wednesday: **2:15pm - 7pm

**Thursday: **4pm - 7pm, late for claimation of prize in competition

**Friday: **2:15pm - 7pm



### Overall schedule of the deliverables based on SSEF submission

As I have been slow and still at the initial task, I don't think I can finish the planning of the whole schedule right now. So I include the discussion for this in the tentative plan and list down the deadline for SSEF registeration.

### General Timeline of SSEF

#### Before registering

* Complete Project Abstract
* Complete Research Plan
* Confirm Project Title
* Confirm Project Category and Sub-Category

#### Process

* Online registeration by **4th Nov** (Registeration open from **1st Nov**)
* Inform teachers of registration through the given link
* Complete SSEF Forms by **3rd Dec** (Advised deadline for obtaining the necessary signature is **2nd Nov**)
* Online submission of SSEF forms by **7th Dec**
* Complete the research project by **31st Dec**
* Completion of "*The Cover Page*" and "*Project Report*"
* Online Submission of Project Docyments by **4th Jan**

Below is the official checklist for SSEF registeration.

![image-20181020174547438](/Users/SunXinyuSingapore/Library/Application Support/typora-user-images/image-20181020174547438.png)



### Tentative meeting plan

As I'm visiting IHPC on Monday, is it possible for us to arrange a meeting please? I'll stay in IHPC from 2:15pm to 7pm, so any time between these is fine to me.

* Discussion about overall schedule:
  * What other problems I'm going to have?
  * The main focus and common features of all tasks
* Feedback regarding SVM model (I'll implement by next Monday)

