# Logistic-Regression-Model

# About

This data set describes operating conditions of a reactor and contains class labels about whether the reactor will operate or fail under those operating conditions. Your job is to construct a logistic regression model to predict the same.
Dataset_Question2.xlsx: The data contains a 1000 X 6 data matrix. The first five columns are the operating conditions of the reactor. The sixth column provides necessary annotation:
● Temperature: 400-700 K
● Pressure: 1-50 bar
● Feed Flow Rate: 50-200 kmol/hr
● Coolant Flow Rate: 1000-3600 L/hr
● Inlet Reactant Concentration: 0.1-0.5 mole fraction
● Test: fail/pass. Whether the reactor will operate or fail under the corresponding operating conditions.
Using the above datasets, make a report on the following: Note: Any assumptions made should be properly mentioned. 
1. Describe the statistics of the data.
2. Partition your data into a training set and a test set. Keep 70% of your data for training and set aside the remaining 30% for testing.
3. Fit a logistic regression model on the training set. Choose an appropriate objective function to quantify classification error. Manually code for the gradient descent procedure used to find optimum model parameters. (Note: You may need to perform multiple initializations to avoid local minima)
4. Evaluate the performance of above model on your test data. Report the confusion matrix and the F1 Score.
