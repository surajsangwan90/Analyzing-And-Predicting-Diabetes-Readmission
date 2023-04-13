                                                    **1.1 - Business Objective**
The problem statement and business objective is a hospital readmission, when a person is discharged from the hospital and gets readmitted again after a certain period of time. In this Data, whoever is first admitted in the hospital due to any problem,is  also suffering from diabetes. It is increasingly recognised that the diabetes in the hospitalized patient has significant effects in terms of mortality and morbidity rate(1). Hospital Readmission rates are indicating the quality of the hospital and several other factors for example, the admitted time in the hospital, number of inpatient visits, emergency visits etc. Hospitalized patients are more likely to get readmitted in the hospital. Therefore, reducing the readmission rates for diabetic patients will save a large amount of medical cost significantly.
My Goal of this problem is to reduce the hospital readmission by finding the factors that lead to higher readmission rates within 30 days after discharge from the hospital . We are using 10 years( 1999-2008) of clinical care at 130 hospitals in the United States(US). I am going to answer these questions below:
What are the strongest predictors of hospital readmission in the Hyperglycemia patient?
 How well can we predict the Hospital Readmission with these given features? 
 
                                                   1.2 - Asses the Situation 
                                                   
The US Government is taking strict action against the Hospital by applying financial penalties of excess readmission of medicare patients(2). Readmission rates have now declined. In 2003 about 20 percent of patients were readmitted to the hospital within 30 days after discharge from the hospital. The Admission rates declined from 21.5% to 17.8 % from 2007 to 2015(3).
Data: This dataset was obtained from the health fact database, a national data warehouse that collects data from the hospitals across the United States. This dataset contains more than 100000 records and 50 features such as age, race, time in hospital, lab procedures, diagnosis and a list of medicines, etc. 
Risk:  This clinical data contains valuable information but heterogeneous and very difficult data in terms of many missing values, skewed data, incomplete records, and very less insights are available. This is a challenging situation and it can hamper my model accuracy.
Contingency:   We will use the different techniques to remove the outliers, impute the missing value, transform the data types, analyze the initial data by using different charts, feature engineering, and will select the appropriate modelling techniques. Since We will be predicting the hospital readmission, so we will use different kinds of classification algorithms. 
Tool: We will use the Jupyter and Pyspark to process the data sets, check the quality of the data and to generate the data models.

                                                  1.3 - Data mining objectives
                                                  
This step has two major parts one is the data mining goals and other one is the success criteria of this goal:

        1.3.1 Data Mining Goal
Here my data mining objective is to reduce the hospital readmission within 30 days after discharge from the hospital. It can be achieved by finding the hidden patterns in my dataset with the help of machine learning algorithms. Here is the data mining goals:
Identifying the high risk predictor variables based on the past data (Profiling)
Use this historical data to predict whether a patient will be readmitted in the hospital within 30 days or not (Classification)
These data mining goals mentioned above, if met, can then be used by the business to reduce the readmission rate by finding the patterns and can save their valuable life.


       1.3.2 Data Mining Success Criteria

In this step, We are going to write down some of the criteria for success. Based on those criteria, We will get to know if our Model is successful or not. These are the criterias mentioned below:


For the Model assessment, we will focus on Accuracy and Recall value.
If the Accuracy for testing and training is above than 70% then I will consider this as a successful Model(5 ). 
For Evaluation Metrics, I will mark Area Under Curve(AUC) matrics.
If the AUC  is greater than 70% then our model is successful(5).
Apart from these matrices, We will find out the F1 Score & Precision Score.
Apart from all of these things, Our Model should not be Overfit.
Based on these above metrics, We will find out if our model is successful or not.

                                                    1.4 - The project plan
The objective of this study is to determine the main factors that lead to higher readmission among diabetic patients and correspondingly being able to predict which patient will likely to get readmitted can improve the quality of life as well.We will use this dataset in the Jupyter and Pyspark to get answer of these questions: 
What are the most important features of hospital readmission among patients with diabetes?
How well can we predict hospital readmission with these limited features only?

