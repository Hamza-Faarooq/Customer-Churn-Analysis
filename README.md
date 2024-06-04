
# Customer Churn Analysis Project

## Project Overview

This project aims to analyze customer churn for a telecommunications company. Customer churn refers to the phenomenon where customers stop using a company's services. The objective is to identify the factors contributing to churn and develop a predictive model to identify customers at risk of churning. Based on the analysis, strategic recommendations will be provided to improve customer retention.

## Project Structure

The project is divided into several stages:

1. **Data Generation and Collection**
2. **Data Preparation**
3. **Exploratory Data Analysis (EDA)**
4. **Model Training and Evaluation**
5. **Results Interpretation and Insights**
6. **Recommendations**
7. **Reporting and Presentation**

## Data Generation and Collection

### Synthetic Data Generation

We generated synthetic data to simulate a realistic dataset for a telecommunications company. The data includes:

- Customer Demographics
- Service Usage Data
- Billing Information
- Customer Support Interaction
- Churn Labels

### CSV Files

The following CSV files were created using the synthetic data:

- `customer_demographics.csv`
- `service_usage.csv`
- `billing_info.csv`
- `customer_support.csv`
- `churn_label.csv`

## Data Preparation

### Steps

1. **Data Integration:** Combine data from various sources into a single dataset.
2. **Data Cleaning:** Handle missing values, outliers, and inconsistencies.
3. **Feature Engineering:** Create new features that might be useful for analysis (e.g., tenure, average usage).
4. **Data Transformation:** Normalize numerical features, encode categorical variables.

## Exploratory Data Analysis (EDA)

### Techniques

- **Descriptive Statistics:** Mean, median, mode, variance, standard deviation.
- **Visualization:** Histograms, box plots, scatter plots, correlation heatmaps.
- **Segmentation Analysis:** Grouping customers by similar characteristics (e.g., high usage vs. low usage).

## Model Training and Evaluation

### Approach

1. **Model Selection:** Compare multiple models (e.g., Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, Neural Networks).
2. **Training and Validation:** Split the data into training and validation sets.
3. **Hyperparameter Tuning:** Use Grid Search or Random Search to find the best parameters.
4. **Model Evaluation:** Assess models using accuracy, precision, recall, F1-score, ROC-AUC.

## Results Interpretation and Insights

### Analysis

- **Feature Importance:** Identify which features have the most significant impact on churn.
- **Customer Profiles:** Describe typical profiles of customers who churn vs. those who stay.
- **Predictive Power:** Evaluate the effectiveness of the predictive model.

### Visualizations

- Feature importance plots
- Confusion matrix
- ROC curves

## Recommendations

Based on the insights derived from the data, strategic recommendations are provided to the company. This may include:

- Improving customer support based on common issues.
- Tailoring marketing efforts to high-risk customers.
- Enhancing services and offers to increase customer satisfaction.

## Reporting and Presentation

### Deliverables

- **Executive Summary:** Key findings and recommendations.
- **Detailed Report:** Methodology, analysis, and results.
- **Presentation Slides:** Highlights of the project for stakeholders.

## Project Code

### Data Generation and Saving to CSV

```python
import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Number of entries
num_entries = 30

# Generate synthetic customer demographics data
customer_ids = np.arange(1001, 1001 + num_entries)
ages = np.random.randint(18, 70, size=num_entries)
genders = np.random.choice(['Male', 'Female'], size=num_entries)
incomes = np.random.randint(30000, 120000, size=num_entries)

demographics = pd.DataFrame({
    'customer_id': customer_ids,
    'age': ages,
    'gender': genders,
    'income': incomes
})

# Generate synthetic service usage data
services_used = np.random.choice(['Internet', 'Phone', 'Both'], size=num_entries)
usage_frequency = np.random.randint(1, 50, size=num_entries)
duration = np.random.randint(1, 100, size=num_entries)

usage = pd.DataFrame({
    'customer_id': customer_ids,
    'services_used': services_used,
    'usage_frequency': usage_frequency,
    'duration': duration
})

# Generate synthetic billing information data
monthly_charges = np.random.uniform(20.0, 150.0, size=num_entries)
total_charges = monthly_charges * np.random.randint(1, 24, size=num_entries)
payment_methods = np.random.choice(['Credit Card', 'Debit Card', 'Bank Transfer', 'E-Wallet'], size=num_entries)

billing = pd.DataFrame({
    'customer_id': customer_ids,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'payment_method': payment_methods
})

# Generate synthetic customer support interaction data
support_calls = np.random.randint(0, 10, size=num_entries)
issues_raised = np.random.randint(0, 5, size=num_entries)
resolutions = np.random.choice(['Resolved', 'Unresolved'], size=num_entries)

support = pd.DataFrame({
    'customer_id': customer_ids,
    'support_calls': support_calls,
    'issues_raised': issues_raised,
    'resolutions': resolutions
})

# Generate synthetic churn data
churn = np.random.choice([0, 1], size=num_entries)

churn_data = pd.DataFrame({
    'customer_id': customer_ids,
    'churn': churn
})

# Save all dataframes to CSV files
demographics.to_csv('customer_demographics.csv', index=False)
usage.to_csv('service_usage.csv', index=False)
billing.to_csv('billing_info.csv', index=False)
support.to_csv('customer_support.csv', index=False)
churn_data.to_csv('churn_label.csv', index=False)
