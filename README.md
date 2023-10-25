# Machine Learning Capstone Project:
A capstone project for Data Scientist Bootcamp Certification.

# Project Scope:
Building the Machine Learning Model.

# Project Problem Statement
A credit card is one of the most used financial products to make online purchases and payments, which should be safe from fraud. Hence, is important that credit card companies recognize fraudulent credit card transactions so that customers are not charged for items they did not purchase.

# Project Objective:
To build a Machine Learning classification model with a Classifier to predict whether a creditcard transaction is fraudulent or not. The project aims at testing the personal skills in Machine Learning Model building aiming at building a classifier with a predictive power with accuracy above 75%.

# Programming Language used:
Python language is used throughout the project.

# Project Methodology:
## To Accomplish the Project Below are the Methodological Steps to Follow:
* Data Collection.
* Exploratory Data Analysis: Analyze and understand the data to identify patterns, relationships, and trends in the data by using Descriptive Statistics and Visualizations.
* Data Cleaning: This might include standardization, handling the missing values and outliers in the data.
* Dealing with Imbalanced data: This data set is highly imbalanced. The data should be balanced using the appropriate methods before moving onto model building.
* Feature Engineering: Create new features or transform the existing features for better performance of the ML Models.
* Model Selection: Choose the most appropriate model that can be used for this project.
* Model Training: Split the data into train & test sets and use the train set to estimate the best model parameters.
* Model Validation: Evaluate the performance of the model on data that was not used during the training process. The goal is to estimate the model's ability to generalize to new, unseen data and to identify any issues with the model, such as overfitting.
* Model Deployment: Model deployment is the process of making a trained machine learning model available for use in a production environment.

# Data and Data Source:
## Data Source:
Creditcard records collected for the two-day transactions in Europe whereby the dataset is provided by KnowledgeHud upGrad as one of the choices among many project concepts learners to consider.

## Nature of Data:
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where 492 frauds out of 284,807 transactions were identified.

# Loading of Dataset:
Dataset is loaded by creading a DataFrame 'df' using the Pandas library using the read_csv() method.

## Checking for successful dataset loading using a few methods:
* df.columns.
* df.shape.
* df.sample().
* df.head().
* df.tail().

# Conducting Exploratory Data Analysis (EDA) to see trends and patterns about the dataset features:
## Features pairwise correlation with target 'Class':
* Used sv.analyze() method after importing the library sweetviz as sv.
* Called the stored variable (report) using show_notebook() method.

## Investigating through the DataFrame checking the following:
* Datatypes.
* Detailed DataFrame information using df.info() method where null values are observed (if any).
* Checked for duplicates and unique values.
* Checked for feature values Minimum and Maximum to determine the convenient approach when scaling to within 0 and 1 values using df.describe() method.
* Conducted Time conversion to get the correct time format (from the numeric) to the date-time format using to_datetime() method.
* Visualized the Time column plotting a Histogram to find out the time distribution on transactions.
* Visualized transactions Amount to spot outliers.
* Visualized Transaction Amount vs Target ('Class') using Seaborn library.
* Visualized pairwise correlation across features.
* Visualized the distribution of the Class values using Matplotlib.pyplot as plt (pie chart) to confirm the imbalanced dataset.

# Performing Feature Transformation as Preparation for the Machine Learning Model to Efficiently train:
## Performing Feature Encoding
* Converted Time to binary value 0 and 1 integer datatype after clustering into two
* After all features were in float and integer datatypes, realized all float datatypes were in range above 1 value, hence need for scaling [0 & 1].

## Performing Feature Scaling
Except for Time and Class features, the rest were scaled uning MinMaxScaler() method after importing it from the sklearn.preprocessing library.

## X, y split from df2 (sampled dataset):
Supported by the Numpy Library droped y from the DataFrame using drop() method and storing y as testLabels in the integer datatype format

## Performed X_train, y_train split:
Using train_test_split() method imported from sklearn.model_selection was able to store X_train, y_train, X_test, y_test sets

# Model Selection:
* Given the need for imbalanced dataset handling such that iteration of different approach will be needed trying out to get high model performance metrics, Model predictive power, easy to handle and computation simplist, drove the selection of the Machine Learning Model.
* Long-Short Term Memory (LSTM) Machine Learning, which is a Recurring Neuron Network (RNN) model was preferred over Logistic Regression, Random Forest, and Xgboos models.
* LSTM ability to handle Vanishing Gradient Problem, ability to train efficiently on time series dataset, and with number of epochs flexibility are key LSTM model selection points creteria.
  #### LSTM Model Structure
<img src="https://modeling-languages.com/wp-content/uploads/2019/07/Architecture-EncoderDecoder_v2-1080x453.png">
  Source: https://modeling-languages.com/lstm-neural-network-model-transformations/

# Building the Model:
## Importing the Important Libraries to support the model training and testing/evaluating, reporting, and visualization
* from keras.models import Sequential
* from keras.layers import LSTM, Dense
* from keras.optimizers import Adam
* from sklearn.metrics import confusion_matrix, classification_report
* import seaborn as sns
* import matplotlib.pyplot as plt

## Building a callable functional Long-Short Term Memory Mode to predict whether creditcard transactions are fraudulent or not:
* A function of the LSTM model containing:
  #### Pipeline
  * The number of LSTM units, learning rate, and batch size.
  * Sequential,
  * Dense: activation made using sigmoid for a binary classifier.
  * Model optimizer using 'Adam' at 0.001 learning rate.
  * Compiling the Model where loss is equal to the binary cross entropy, with the above optimizer, and accuracy being the central model metric.
* A function to handle the training and model testing/evaluating/validation on unseen dataset.
  ### Pipeline
  * LSTM Model defined in the basis of X_train, y_train, X_test, y_test, and apochs = 'n' where n = number of preferred epochs running the model.
  * Storing the model history by fitting LSTM model with 64 batche size.
  * Storing the y_predictions using the predict() method on the unseen dataset.
  * Deciding on the prediction thredshold whereby y_predictions are above 0.5.
* A function to handle confusion matrix plotting for proper visualization of the model power to predict Class nature.
* A portion of the function to cover calculation of model performance metrics.
* A portion of the function to print the classification report to see the Model Precision, Recall, F1-Score, and Accuracy
* A line to return the y predictions.

## Calling the y predictions:
* Users may change the number of epochs in maximizing the predictive power of the model in terms of performance metrics
* Note further that, wih less number of epochs, the first y_preds given zeros (0) on the fraudulant class (1) because of the missing fraudulent incendencies. This requires then handling of the imbalanced datasets.

### Handling the Imbalanced Dataset
* Given that there is near to zero values on the fraudulent class (1), undersampling technique was dropped in favour of Oversampling and Sythetic Minority Oversampling technique (SMOTE)
* Given the visualization of the confusion matrix and classification report, SMOTE method was preferred.
* SMOTE() method was applied on the undersample (1 Class) after importing it from the imblearn.over_samplling
* Resampled values then applied to update the y_preds. Both the general classification report and confusion matrix provided plausible consistent metrics about the model.

#### Importance of 'SMOTE' over 'oversampling' technique:
* Preservation of Information: SMOTE generates synthetic samples that preserve the original information present in the minority class, reducing the risk of overfitting, which can occur with the traditional oversampling technique.
* Reduced Overfitting: Traditional oversampling methods duplicate existing samples while SMOTE creates new and diverse samples that help in reducing the risk of overfitting.
* Improved Generalization: SMOTE encourages the model to generalize better by providing more diverse examples of the minority class leading to a more robust model that performs well on unseen data.
* Better Boundary Learning: SMOTE helps the model learn the decision boundary between classes more effectively, as it introduces samples along the class boundary. This can lead to a model that makes more accurate predictions.
* Balanced Dataset: SMOTE aims to balance the class distribution while avoiding excessive oversampling. In contrast, traditional oversampling could produce a too large and computationally expensive dataset.
* Reduced Bias: SMOTE reduces the bias introduced by oversampling because it generates synthetic samples based on the distribution of the minority class, ensuring that the synthetic samples represent the true characteristics of the minority class. Wider

# Model Usability:
* Potential for Cost Savings: Using this model by the Bank may save money in preventing fraudulent transactions. It can also improve customer satisfaction by reducing the number of legitimate transactions incorrectly flagged as fraudulent.
* Model Resilience and reliability. This model may maintain its fraudulent predictive power (performance) on unseen data, which is important in a real-world setting where  transactions distribution changes over time.
* Monitoring and Adaptation: Even with a high-performing model, it's essential to continuously monitor its performance and adapt it as needed. Fraudsters can change their tactics, and the model should be updated to address evolving threats related to creditcards' transactions.
* Customer Communication: In the banking domain, clear communication with customers is essential. Customers should be informed about the bank's fraud detection methods and how they can protect themselves. An overzealous fraud detection system may sometimes lead to customer inconvenience.
* Marketing Strategy: The bank may use the high-performance model promoting itself as a secure place to conduct financial transactions, emphasizing the commitment to protecting customers from fraud.
* Regulatory Compliance: Ensure that the model complies with relevant regulations and data privacy laws. Transparency in model development and usage is critical.
* Education and Training: Continuously educating and training the Bank employees on how to use and interpret the model results, as well as how to take appropriate actions when fraud is suspected.
