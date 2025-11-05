# Detection Faults in machines using CV
### Description
In this project, acoustic signals captured from the machines are processed and fed into the CNN model. The CNN learns to extract relevant features from the acoustic data and classify them into different categories, such as normal operation or specific fault types.

By training the CNN model on a labeled dataset containing examples of both normal machine operation and various types of faults arching,corona,losseness,tracking, the model learns to distinguish between different acoustic patterns associated with each condition.plotting the ROC curve and calculating the confusiion matrix.evaluating the model and show the 87% of accuracy.

|Sr#| Topic | 
|-|-|
|00| Overview |
|01| Dataset |
|02| Setup Instruction |
|03| Usage |
|04| Acknowledgment |
 ## Overview
 This project aims to detect faults in machines using a Convolutional Neural Network (CNN) model trained on acoustic dataset. By analyzing acoustic signals captured from machines, the CNN model can identify potential faults or anomalies, enabling proactive maintenance and minimizing downtime.
 ## Dataset
 The dataset used in this project consists of acoustic signals recorded from machines during normal operation and various fault conditions. It includes labeled 
 examples of different fault types, allowing the CNN model to learn to distinguish between normal and faulty machine behavior.
  ## Setup Instruction
Libraries: Make sure you have all the required dependencies installed. You can find the list of dependencies in the requirements.txt file.

Dataset: Download the acoustic dataset and organize it according to the provided directory structure. Ensure that the dataset is split into training, validation, and testing sets.

Training: Train the CNN model using the provided training script. Adjust hyperparameters and network architecture as needed.

Evaluation: Evaluate the trained model's performance on the validation and testing sets to assess its accuracy and generalization capability.

 ## Usage
 Training: Use the training script to train the CNN model on the acoustic dataset.
###### model_with_dropout.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
Evaluation: Evaluate the trained model's performance on the validation or testing set.
###### test_loss, test_acc= model_with_dropout.evaluate(X_test, y_test)
###### Predict probabilities for each class
y_probabilities = model_with_dropout.predict(X_test)

###### Convert probabilities to class predictions
y_pred = np.argmax(y_probabilities, axis=1)

## Acknowledgment
This project was inspired by the need for proactive maintenance in industrial settings.
# Heart Disease Predication Using Machine Learning(Semester Project)
### Description
In this Project we used the Machine Learing Model to predict the Heart Disease. We used the Logistic Regression and KNN classifier to predict the Heart Disease rate.
## Objective
The main objective of developing this project are: 
1.	To develop machine learning model to predict future possibility of heart disease by implementing Logistic Regression. 
2.	To determine significant risk factors based on medical dataset which may lead to heart disease.  
3.	To analyze feature selection methods and understand their working principle. 
### Dataset
The dataset is publicly available on the Kaggle Website at [4] which is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. It provides patient information which includes over 4000 records and 14 attributes. The attributes include:  age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting, sugar blood, resting electrocardiographic results, maximum heart rate, exercise induced angina, ST depression induced by exercise, slope of the peak exercise, number of major vessels, and target ranging from 0 to 2, where 0 is absence of heart disease. The data set is in csv (Comma Separated Value) format which is further prepared to data frame as supported by pandas library in python.
### Conclusion 
The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications, which can be a great milestone in the field of medicine. This project resolved the feature selection i.e. backward elimination and RFECV behind the models and successfully predict the heart disease, with 85% accuracy. The model used was Logistic Regression.  Further for its enhancement, we can train on models and predict the types of cardiovascular diseases providing recommendations to the users, and also use more enhanced models. 





