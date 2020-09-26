The purpose of this project is to identify the presence of heart disease based on certain attributes described below.

The dataset used in this project contains the following information:

Used 303 records of the following attributes to determine the presence of the disease:
Age
Sex
ChestPain
RestBP - Resting BP
Chol - Serum Cholestrol
Fbs - Fasting Blood Sugar
RestECG - Resting ECG
MaxHR - Maximum Heart Rate
ExAng - Exercise induced Angina
Oldpeak - ST depression induced by exercise relative to rest
Slope - Slope of the peak exercise ST segment
Ca - Number of major vessels colored by flourosopy
Thal - Thalassemia
AHD - Presence of Heart Disease

The file code.py contains the following code:

Used the above dataset and removed all examples with missing data and split them into
70-30 train test data. 

Fitted Support Vector Classifier (Linear SVM) to the training data and reported training set and test set accuracy. 

Changed the value of C and plotted the training and test set accuracy with varied values of C. 

Experimented with other Kernels and reported their accuracy on Test data.