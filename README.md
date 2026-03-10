# retail-ml-pyspark-project

 "COMPANY": CODTECH IT SOLUTIONS

 "NAME": THAKOR SUNIL T.

 "INTERN ID": CTIS6330

 "DOMAIN" : DATA ANALYTICS

 "DURATION" : 12 WEEKS 

 "MENTOR" : NEELA SANTOSH

 # Retail Transaction Prediction using PySpark

## Project Overview
This project demonstrates scalable data processing and machine learning using PySpark. The goal is to analyze retail transaction data and predict transaction value using regression models.

## Dataset
Online Retail Dataset containing transaction records including:
- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

## Data Cleaning
The following preprocessing steps were performed:
- Removed missing CustomerID values
- Removed duplicate rows
- Filtered negative quantities and prices

## Feature Engineering
A new feature called **TotalAmount** was created:

TotalAmount = Quantity × UnitPrice

## Machine Learning Models
Two regression models were trained:

1. Linear Regression
2. Random Forest Regression

## Model Evaluation

| Model | RMSE |
|------|------|
| Linear Regression | 296.58 |
| Random Forest | 312.83 |

Linear Regression performed better due to the linear relationship between Quantity, UnitPrice, and TotalAmount.

## Technologies Used
- Python
- PySpark
- Machine Learning
- Data Cleaning
- Feature Engineering

## Key Learning
This project demonstrates scalable data processing using PySpark and the implementation of machine learning models for predictive analytics.

 <img width="944" height="852" alt="Image" src="https://github.com/user-attachments/assets/9445f98a-a190-4f8f-8d49-6fe1328fb78c" />

<img width="946" height="767" alt="Image" src="https://github.com/user-attachments/assets/683adb24-15d1-444f-b337-53e9637dd7fb" />
<img width="628" height="582" alt="Image" src="https://github.com/user-attachments/assets/95a5b49d-fc98-4a15-b444-69786da6fd78" />
<img width="901" height="454" alt="Image" src="https://github.com/user-attachments/assets/9ace8208-dbe0-4f25-98d4-1b029a701bd5" />

<img width="959" height="779" alt="Image" src="https://github.com/user-attachments/assets/ae268182-8b23-46c9-87d9-6f762aeeca79" />
<img width="535" height="571" alt="Image" src="https://github.com/user-attachments/assets/c1aabdd9-eaea-48fa-a7b6-c4bf45ac4b17" />
