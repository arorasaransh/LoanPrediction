#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:41:41 2020

@author: saransharora
"""

"""
Data and Description

Loan_ID: Unique Loan ID
Gender: Male/ Female
Married: Applicant married (Y/N)
Dependents: Number of dependents
Education: Applicant Education (Graduate/ Under Graduate)
Self_Employed: Self employed (Y/N)
ApplicantIncome: Applicant income
CoapplicantIncome: Coapplicant income
LoanAmount: Loan amount in thousands
Loan_Amount_Term: Term of loan in months
Credit_History: credit history meets guidelines
Property_Area: Urban/ Semi Urban/ Rural
Loan_Status: Loan approved (Y/N)
"""


"""
Before starting to code, I am trying to come up with a few Null Hypothesis:
    1. Applicant Income - Higher the income of the applicant, higher will be the chance of loan approval
    2. Loan Amount - Higher the loan amount, lesser will be the chance of getting the loan approved
    3. Loan_Amount_Term - Higher the loan amount term, lesser will be the chance of getting the loan approved
    4. Credit_History: People who have repaid their previous loans, have a higher chance of getting their loan approved
    5. Property_Area: If property in urban area, the chance of getting the loan getting approved should be higher
"""
#Importing dependencies
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('Desktop/AnalyticsVidhya/LoanPrediction/train.csv')
print(data.head())

#checking size
print(data.size)

#checking shape
print(data.shape)

#checking columns
print(data.columns)

#Viewing all columns for the first row
print(data.iloc[0])

#Checking datatypes
print(data.dtypes)  #Object means categorical variables. 

#checking the description and summary
print(data.describe())
print(data.info)

#check for missing values in dataframe
print(data.isnull().sum())

#Making a copy of the train dataset
loan = data.copy()

#Uni-variate analysis
print(loan.Loan_Status.value_counts())
print(loan.Loan_Status.value_counts(normalize=True))

#Visualising Nominal(Categorical) variables
print(loan.Loan_Status.value_counts().plot.bar())
print(loan.Gender.value_counts().plot.bar())
print(loan.Married.value_counts().plot.bar())
print(loan.Self_Employed.value_counts().plot.bar())
print(loan.Credit_History.value_counts().plot.bar())


#Visualising Ordinal(Categorical) variables
print(loan.Dependents.value_counts().plot.bar())
print(loan.Education.value_counts().plot.bar())
print(loan.Property_Area.value_counts().plot.bar())

#Visualise contiuous variables - numerical
AppInc = loan.ApplicantIncome
AppInc.plot.hist(bins=100)
LoanAm = loan.LoanAmount
LoanAm.plot.hist(bins=100)


#Bi-variate analysis
#Checking loan status for Gender
gender_ct = gender_ct.div(gender_ct.sum(1).astype(float), axis=0)
gender_ct.plot.bar(stacked=True) #The proportion of applications approved for male and female is approximately the same

#Checking 