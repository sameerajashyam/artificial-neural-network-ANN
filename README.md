
# Customer Churn Prediction with ANN

This project involves predicting customer churn using an Artificial Neural Network (ANN). The dataset used is the **Churn_Modelling.csv**, which contains information about customers of a bank and whether they have churned (Exited) or not.

## Table of Contents

- Project Overview
- Dataset
- Requirements
- Project Structure
- Usage
- Model Architecture
- Training
- Saving and Loading Models
- TensorBoard

## Project Overview

The project aims to build a predictive model that can identify customers who are likely to churn. The model is built using a neural network with two hidden layers.

## Dataset

The dataset contains the following columns:
- RowNumber: Index of the row.
- CustomerId: Unique identifier for a customer.
- Surname: Customer's surname.
- CreditScore: Customer's credit score.
- Geography: Country of the customer.
- Gender: Gender of the customer.
- Age: Customer's age.
- Tenure: Number of years the customer has been with the bank.
- Balance: Account balance.
- NumOfProducts: Number of products the customer has.
- HasCrCard: Does the customer have a credit card? (1 = Yes, 0 = No).
- IsActiveMember: Is the customer an active member? (1 = Yes, 0 = No).
- EstimatedSalary: Estimated salary of the customer.
- Exited: Did the customer churn? (1 = Yes, 0 = No).

## Requirements

To run this project, you need the following Python libraries:

- pandas
- scikit-learn
- tensorflow

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── Churn_Modelling.csv       # Dataset
├── label_encoder_gender.pkl  # Label Encoder for Gender
├── onehot_encoder_geo.pkl    # OneHot Encoder for Geography
├── scaler.pkl                # Standard Scaler for features
├── model.h5                  # Trained ANN Model
├── logs/                     # TensorBoard logs
└── README.md                 # README file
```

## Usage

1. **Load and Preprocess the Dataset:**
   - Drop irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
   - Encode categorical variables (`Gender`, `Geography`).
   - One-hot encode the `Geography` column.
   - Save the encoders for future use.

2. **Split the Dataset:**
   - Divide the dataset into training and testing sets.
# artificial-neural-network-ANN
