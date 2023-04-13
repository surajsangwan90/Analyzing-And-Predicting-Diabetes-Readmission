# Business Undersatnding


## 1.1 -  Business Objective
 The business objective of this project is a diabetic hospital readmission, when a person is discharged from the hospital and gets readmitted again after a certain period of time. In this Data, whoever is first admitted in the hospital due to any problem,is  also suffering from diabetes. It is increasingly recognised that the diabetes in the hospitalized patient has significant effects in terms of mortality and morbidity rate(1). Hospital Readmission rates are indicating the quality of the hospital and several other factors for example, the admitted time in the hospital, number of inpatient visits, emergency visits etc. Hospitalized patients are more likely to get readmitted in the hospital. Therefore, reducing the readmission rates for diabetic patients will save a large amount of medical cost significantly.
My Goal of this problem is to reduce the hospital readmission by finding the factors that lead to higher readmission rates within 30 days after discharge from the hospital . We are using 10 years( 1999-2008) of clinical care at 130 hospitals in the United States(US). I am going to answer these questions below:
What are the strongest predictors of hospital readmission in the Hyperglycemia patient?
 How well can we predict the Hospital Readmission with these given features? 
 
## 1.2 - Asses the Situation 
                                                   
The US Government is taking strict action against the Hospital by applying financial penalties of excess readmission of medicare patients(2). Readmission rates have now declined. In 2003 about 20 percent of patients were readmitted to the hospital within 30 days after discharge from the hospital. The Admission rates declined from 21.5% to 17.8 % from 2007 to 2015(3).
Data: This dataset was obtained from the health fact database, a national data warehouse that collects data from the hospitals across the United States. This dataset contains more than 100000 records and 50 features such as age, race, time in hospital, lab procedures, diagnosis and a list of medicines, etc. 
Risk:  This clinical data contains valuable information but heterogeneous and very difficult data in terms of many missing values, skewed data, incomplete records, and very less insights are available. This is a challenging situation and it can hamper my model accuracy.
Contingency:   We will use the different techniques to remove the outliers, impute the missing value, transform the data types, analyze the initial data by using different charts, feature engineering, and will select the appropriate modelling techniques. Since We will be predicting the hospital readmission, so we will use different kinds of classification algorithms. 
Tool: We will use the Jupyter and Pyspark to process the data sets, check the quality of the data and to generate the data models.

## 1.3 - Data mining objectives
                                                  
This step has two major parts one is the data mining goals and other one is the success criteria of this goal:

### 1.3.1 Data Mining Goal
Here my data mining objective is to reduce the hospital readmission within 30 days after discharge from the hospital. It can be achieved by finding the hidden patterns in my dataset with the help of machine learning algorithms. Here is the data mining goals:
Identifying the high risk predictor variables based on the past data (Profiling)
Use this historical data to predict whether a patient will be readmitted in the hospital within 30 days or not (Classification)
These data mining goals mentioned above, if met, can then be used by the business to reduce the readmission rate by finding the patterns and can save their valuable life.


### 1.3.2 Data Mining Success Criteria

In this step, We are going to write down some of the criteria for success. Based on those criteria, We will get to know if our Model is successful or not. These are the criterias mentioned below:


For the Model assessment, we will focus on Accuracy and Recall value.
If the Accuracy for testing and training is above than 70% then I will consider this as a successful Model(5 ). 
For Evaluation Metrics, I will mark Area Under Curve(AUC) matrics.
If the AUC  is greater than 70% then our model is successful(5).
Apart from these matrices, We will find out the F1 Score & Precision Score.
Apart from all of these things, Our Model should not be Overfit.
Based on these above metrics, We will find out if our model is successful or not.

## 1.4 - The project plan
The objective of this study is to determine the main factors that lead to higher readmission among diabetic patients and correspondingly being able to predict which patient will likely to get readmitted can improve the quality of life as well.We will use this dataset in the Jupyter and Pyspark to get answer of these questions: 
What are the most important features of hospital readmission among patients with diabetes?
How well can we predict hospital readmission with these limited features only?

# Data Understanding


## 2.1 Collect Initial Data
I have collected the data of 10 years(1999-2008) at 130 hospitals in the United States  from the centre for machine learning and Intelligent systems.(4)
https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
During the extraction of data from UCI Machine learning repository, I did not face any kind of issues.The following information is extracted from the repository: 
Diabetes readmission data file: which has all the features information of the dataset
Ids Mapping: which has information about some fields

## 2.2  Data description
There are a total 101766 records of diabetic patients in this dataset and there are a total 50 features including one target variable available. 

## 2.3- Explore the Data 
We are using Pyspark and Jupyter Notebook to explore and visualize the data. There are a total of 101766 valid records and 50 fields.  The data types of the fields are string types. The target variable is readmitted, which has three unique values, and the rest of the 49 fields are the predictor variable. 
There are many blank values in the dataset, but they are not showing up in the data because they are imputed in the data by the question mark “?” sign. We will later identify these values and then take action accordingly.

# 2.4- Data Quality
## 2.4.1 Missing Values:
While exploring the data, I have observed that there are many outliers and extremes in this dataset. As I discussed earlier, there are many missing values in the dataset, but these are imputed by the "?” sign. So I have considered this sign “?” as a missing value.

## 2.4.2 Outliers:

As shown in the table below, there are many outliers in the data, including discharge disposition, admission source, num procedures that have the maximum number of outliers. There are many methods to find outliers in the data. We can find out the outliers by using PySpark, and through the visualization we can also find out the outliers. Later We will remove these outliers 

# Step 3. Data Preparation

Data preparation is the heart of data mining. It is a very tedious and time consuming process. It is estimated, it takes approx 40-60 % of the project’s time and effort.  Data preparation typically involves the following steps:
## Selecting the Data
## Clean the Data
## Construct the Data
## Integrate various data source as required
## Format the Data as required

# Step 4. Data Transformation

Data transformation is basically divided into two sub parts, one is data reduction and the other one is data projection.

## 4.1  Data Reduction
In this step, We will reduce the size of the data attributes by using the important features selection. As compared to pandas, pyspark has different kind of features reduction methods like PCA, and singular value decomposition(9), but I really find these methods hard to interpret, because there are many attributes which are really very important, so I have done this step manually to remove the unimportant variables from the dataset. Later, In step 4.2, I have used the features selection method. 
We already have removed some unimportant attributes in step 3.1. As we can see in the output file of the code, the citrogliptone, and the examide column has only 1 unique value which can affect my model. To avoid biaseness, We have dropped these columns from the dataset. Just to show the unique values, I have converted the pyspark dataframe into the pandas dataframe, but we have not kept this pandas dataframe further.
As we can also see from output of the code, the examide, and the citrogliptone have only 1 value (No). These two attributes are drug types in the data, whether a patient has taken this drugs or not, so there is no possibility to find out the reason why people are readmitted within 30 days with these attributes.

## 4.2  Data Projection

In this step, we will add some important columns that will help us to build a good model. First of all we have imported the stringindexer which converts the categorical data into the numerical one, because most of the models understand the numbers rather than the text, and it will also help us in feature transformation as well. With the help of stringindexer, we have converted all the categorical columns into the numerical columns as shown in figure 24(output in the code), and then we have transformed these columns.

As we can see from figure 25(output in the code), we have imported the VectorAssembler. It is a transformer that combines a given list of columns into a single vector column(10). Vectorassembler is very useful for rawa features and features generated by different feature transformers as well into a single feature vector, in order to train machine learning models like decision tree, logistic regression, etc. In each row, the values of input columns concatenate into a vector in the order.  It accepts all types of variables.
As shown in figure 25(output in the code), We have combined all the features into a single vector column called features and use it to predict the readmission rate  in the hospital

Unlike Pandas, instead of using all the columns together, we just have to use only 1 column to predict the target variables. 

As shown in figure 26(output in the code), we have imported the StandardScaler from pyspark.ml.features. StandardScaler transforms a dataset of vector rows into a standard, so that the mean is always equal to zero and normalizes each feature to have unit standard deviation(10).

So now we have converted the features  column into a scaled_features.  We will use this column into the machine learning algorithms.

## Data Balancing:
Since my dataset is very imbalanced, the number Ones are only 10% approximately and the number of zeroes are 90%. So either we can use under sampling, or we can use oversampling to balance our dataset. 
I have used here the oversampling method to create duplicates record of 1, we have imported the pyspark.sql.function here, by using this function we have duplicated the minority class records. As shown in figure 27, after oversampling, the number of ones is approximately 49%. That is fair enough.


# Step 5. Data Mining Method














## 5.1  DM Methods

## Data Mining: 
Data mining is a process of extracting meaningful information from large amounts of data sets. The gap between data and the business is reduced by using a large amount of machine learning algorithms.  It can be referred to as KDD or knowledge discovery from data.  There are different kinds of data mining methods available as per the business problem. 
There are many methods used for data mining but the crucial part is to select the appropriate method based on our business goal. These methods will help us to predict the future and make business decisions accordingly.

In this business problem, We have to predict whether a patient will be readmitted in the hospital  in future or not based on their medical history. 
As we already know our target variable, therefore we will use the Supervised Learning  here.

**Supervised Learning**
Supervised machine learning is used where we have a label or target variable in the given dataset. It allows collecting data and then produces an output from the previous experience. In Supervised learning, there are some common data mining methods used like Regression,Classification, Decision Tree, etc. 
Regression method is used when we have a continuous type output, for example:  to predict a house price, stock rate prediction.
Classification, prediction and Decision Tree is used when having a discrete output. Therefore, We will use these data mining methods mentioned below(6)

**Classification Method
**Prediction Method
**Decision Tree Method**




![Supervised Machine learning works](https://medium.com/@jorgesleonel/supervised-learning-c16823b00c13)
                                    Figure 28: 


**Classification Method:**  Classification is to classify something according to the shared qualities or characteristics. It means arranging the mass of data into different classes based on the shared qualities and resemblances. These are the parts of supervised learning algorithms.  

Example: Animal Classification 
Cats, dogs, Rabbits are classified as Mammals based on their similarities.
Snake,Crocodile are classified as Reptiles based on their similarities.

**Prediction Method:**  This method is used to predict the future based on the historical data and trends.

Example: House prediction, stock rate prediction, coronavirus cases prediction

**Decision Tree:** A decision tree is a flow chart and in the form of a tree. It is made of different nodes with child nodes. Where the top most node is called a root node. 
These trees are used to analyze different potential outcomes, costs, and consequences of a complex decision. 


## 5.2  DM Methods with data mining goals

Since We have to predict whether the patient will be readmitted to the hospital or not.

Classification Method will be used here because we have to predict the class , Readmitted class(1) or Otherwise class(0). Based on the shared qualities, we will be predicting the class of an encounter. The better will be the prediction, the robust will be our model. 

Prediction Method is used to predict the future based on the past data. We are having 10 years of Hospital readmission data with a huge number of records. By using this method, We can find the future outcomes and can prevent readmission in the hospital.

Decision Tree  can help us to find the top predictors which are contributing to the target variables. Like in this hospital readmission data, we will find out the root cause of the problem and we can prevent it by changing our strategies.

These all methods will help us to decline  the Hospital readmission rate. 


# Step 6. Data Mining Algorithm Selection

## 6.1 Exploratory Analysis of Data-Mining Algorithms concerning DM Objectives 

In this section, we will discuss the different types of supervised learning algorithms. There are many supervised algorithms available in machine learning. We have to choose the few of them as per the project details(7):
Logistic Regression Algorithm
Random Tree Algorithm
Decision Tree Algorithm
Gradient Boosting Algorithm

### 6.1.1 Logistic Regression Algorithm:
It is a supervised classification algorithm. It is again used as a classifier, but it is strictly used for the binary classification either 0 class or 1 class.
We have applied this algorithm and generated a model. I got 61% accuracy with this model as shown in figure 29.

### 6.1.2 Random Tree Algorithm:
Random forest is a classifier, It builds decision trees on different samples and takes their majority vote for classification. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.We have applied this algorithm and generated a model. I got 62% accuracy with this model as shown in Figure 30.

### 6.1.3 Decision Tree Algorithm(CART):
The goal of using a Decision Tree algorithm is to create a model that will predict the class of the target variable by learning simple decision rules that are inferred from the training data. Here we are using a Classification And Regression Tree(CART), it is required to build a decision tree based on the Gini’s impurity index. CART uses the gini index as a measure of the impurity. 
We have applied this algorithm and generated a model. It gives 61% accuracy as shown in figure 31.

### 6.1.4 Gradient Boosting Algorithm:
XGBoost stands for extreme gradient boosting. It is a decision tree algorithm, and It provides parallel tree boosting and is the leading machine learning library for regression, and classification problems. Here We will use these algorithms for classification purposes only .
We have applied this algorithm and generated a model. I got 63% accuracy with this model as shown in figure 32

## 6.2 Selecting Algorithms(s) based on Discussion and Exploratory Analysis 
In the previous step, we have used a total of four algorithms( Decision Tree Classifier, Random Forest, gradient boosting,and logistic regression algorithms).  We also analyzed the evaluation report for each algorithm, which gave the precision score, recall, f1 score, and the accuracy. Which is one of the most important criteria for our model.
We are primarily interested in accuracy and AUC here as discussed in step 1.3 earlier. So based on the accuracy and AUC, we will select the one model out of the given model and that will be our final model for further analysis. Below are the analysis of these algorithms:

**In the CART Algorithm, The  accuracy of the model is 61% and AUC is 62%. 
In the Random Tree Algorithm, The accuracy of the model is 62% and AUC is 66% . 
In Gradient Boost Algorithm, The accuracy of the model is 63% and AUC is 68% . 	 
In the Logistic Regression Algorithm, The accuracy of the model is 61% and AUC is 65%.	**

In the above models, Gradient Boost Algorithm has 63% accuracy and 68 % AUC score, while other Algorithms have less accuracy and AUC Score. So the winner is the Gradient Boost Algorithm.


## 6.3. Build/Select Model with Algorithm/Model Parameter(s) 
As we already discussed in the previous step, Gradient Boosting is my final model. 
Gradient boosting is one of the boosting algorithms; it is used to minimize bias error of the model. It is used for the Classification problem. It is a sequential technique which works based on the ensemble principle. It combines a set of weak learners and delivers an improved prediction accuracy.
We have used the default parameters so far. 
As shown in figure 33, we have imported the GBTClassifier and the parameters I have chosen, the scaled_features of the dataset, and the target variable, and all other parameters are defaults. 
Then we have fitted the model with the parameters set above with training data. After training the model we will predict the output by using testing data without target variable in testing data. 
We were able to draw the confusion matrix, which is a very crucial part of any machine learning model. Also, we have also plotted the ROC Curve based on the AUC value.

# Step 7. Conclusion
We have successfully completed Multiple iterations so far, and the winner is the iteration with the hyperparameter max depth  to 12. It also meets my data mining success criteria defined in step 1.3.  




## Git commands used for push code into the repository:
**Git init- for initialization. 
**Git add** <file name> or git add . (for all files added
Git status- to check the status of the file
Git commit -m “message”- to commit the changes into stages
Git push- to push the selected files**

  
  
 # References
G. E. Umpierrez, S. D. Isaacs, N. Bazargan, X. You, L. M. Thaler, and A. E. Kitabchi, “Hyperglycemia: an independent marker of in-hospital mortality in patients with undiagnosed diabetes,” Journal of Clinical Endocrinology and Metabolism, vol. 87, no. 3, pp. 978–982, 2002.
Centers for Medicare & Medicaid Services. Readmissions Reduction Program. Available at: http://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Readmissions-Reduction-Program.html (Accessed on December 11, 2012).
https://www.uptodate.com/contents/hospital-discharge-and-readmission/abstract/38
Diabetes 130-US hospitals for years 1999-2008 Data Set https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
How To Know if Your Machine Learning Model Has Good Performance | Obviously AI. (n.d.) from https://www.obviously.ai/post/machine-learning-model-performance
Pedamkar, P. (2021, October 16). Data Mining Methods. EDUCBA. from https://www.educba.com/data-mining-methods/?source=leftnav
Education, I. C. (2021, June 30). Supervised Learning. from https://www.ibm.com/cloud/learn/supervised-learning
Gholamy, A. (n.d.). Why 70/30 or 80/20 Relation Between Training and Testing Sets: A Pedagogical Explanation. ScholarWorks@UTEP. from https://scholarworks.utep.edu/cs_techrep/1209/
Dimensionality Reduction - RDD-based API - Spark 3.3.0 Documentation. (n.d.), from https://spark.apache.org/docs/latest/mllib-dimensionality-reduction
Extracting, transforming and selecting features - Spark 3.3.0 Documentation. (n.d.), from https://spark.apache.org/docs/latest/ml-features.html
Leonel, J. (2019, October 10). Supervised Learning - Jorge Leonel. Medium, from https://medium.com/@jorgesleonel/supervised-learning-c16823b00c13
ML Pipelines - Spark 3.3.0 Documentation. (n.d.). from https://spark.apache.org/docs/latest/ml-pipeline.html
Raj, Saurabh. How to Evaluate the Performance of Your Machine Learning Model. From https://www.kdnuggets.com/2020/09/performance-machine-learning-model.html




