# Progress Summary 2018/12/03

## What's New

* Adding the new features (location and station index) into the dataset and make necessary modification
* I've relooked at my model in the past and tried some figures like MAE and RMSE and realised that the past models ar terribly inaccurate.. 
* After adding in the new features, I'm right now running the program and changed the measure from MAPE (which I often get rediculously large scores, ie over 200+) to more common measures like MAE, MSE and RMSE.
* Also I realise that I can simply convert night data to be zero rather than remove them or keep them as NAN, so I just redo the time series model and get a MAE around 0.9, the problem about the time series model is that it can only accept the continuous CSR input rather than a combination of time, location and CSR, which makes it less accurate, I'll talk more in my final report.
* Also while waiting for the svm model to complete its running (which is still running until the time I send the email), I read something about OneVsAll Classifier, which, suggested by a stack overflow user, can improve the speed of an svm model without losing much accuracy. I've implemented that and still waiting for it to finish now.. (also haven't finished yet)
* As I've mentioned, the models I've built so far are all classifier (except for now the time series model is a regression model), I add in a random forest regressor and a KNN regressor (Ok i admit that i'm just a bit lazy to begin with the report.. cough cough)
* By right random forest classifier model should have high accuracy and fast speed, yet when I really test it, the speed is fast but the accuracy is very low though.. (MAE over 2 and MSE around 16) I'll see whether I have the time to do a gridsearchcbv something after going a full round..

## Schedules to IHPC

All the five days..

as all the commitments are over and it would be best for me to stay with my desktop in IHPC to know how it is going..