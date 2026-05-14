# European Bank Customer Churn Analytics App

A Streamlit web app for **Customer Segmentation & Churn Pattern Analytics in European Banking**.

This project analyzes customer churn patterns using demographic, financial, and behavioral banking data. It includes an interactive churn dashboard, customer segmentation using K-Means clustering, and a machine learning churn prediction model.

## Project Features

- Dataset overview and data quality checks
- Churned vs retained customer analysis
- Country-wise churn rate
- Age group churn analysis
- Product usage vs churn rate
- Balance and credit score comparison by churn status
- Active vs inactive member churn analysis
- K-Means customer segmentation
- Random Forest churn prediction model
- Feature importance analysis
- Business insights and recommendations

## Dataset

The project includes the dataset here:

```text
data/European_Bank.csv
```

Main columns used:

```text
Year
CustomerId
Surname
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
Exited
```

Target column:

```text
Exited
0 = Retained Customer
1 = Churned Customer
```

## How to Run Locally

1. Clone this repository:

```bash
git clone https://github.com/your-username/european-bank-churn-analytics.git
cd european-bank-churn-analytics
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

Use these settings on Streamlit Cloud:

```text
Repository: your GitHub repository
Branch: main
Main file path: app.py
```

## Recommended Repository Name

```text
european-bank-churn-analytics
```

## Project Objective

The goal of this project is to help banks understand customer churn behavior and identify high-risk customer segments for better retention planning.

## Tools Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

## Author

Mohd Zeeshan Khan
