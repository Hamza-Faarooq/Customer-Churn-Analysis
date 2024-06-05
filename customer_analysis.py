import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Generate synthetic data
np.random.seed(42)
num_entries = 30
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

services_used = np.random.choice(['Internet', 'Phone', 'Both'], size=num_entries)
usage_frequency = np.random.randint(1, 50, size=num_entries)
duration = np.random.randint(1, 100, size=num_entries)

usage = pd.DataFrame({
    'customer_id': customer_ids,
    'services_used': services_used,
    'usage_frequency': usage_frequency,
    'duration': duration
})

monthly_charges = np.random.uniform(20.0, 150.0, size=num_entries)
total_charges = monthly_charges * np.random.randint(1, 24, size=num_entries)
payment_methods = np.random.choice(['Credit Card', 'Debit Card', 'Bank Transfer', 'E-Wallet'], size=num_entries)

billing = pd.DataFrame({
    'customer_id': customer_ids,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'payment_method': payment_methods
})

support_calls = np.random.randint(0, 10, size=num_entries)
issues_raised = np.random.randint(0, 5, size=num_entries)
resolutions = np.random.choice(['Resolved', 'Unresolved'], size=num_entries)

support = pd.DataFrame({
    'customer_id': customer_ids,
    'support_calls': support_calls,
    'issues_raised': issues_raised,
    'resolutions': resolutions
})

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

# Load datasets
demographics = pd.read_csv('customer_demographics.csv')
usage = pd.read_csv('service_usage.csv')
billing = pd.read_csv('billing_info.csv')
support = pd.read_csv('customer_support.csv')
churn = pd.read_csv('churn_label.csv')

# Merge datasets
df = demographics.merge(usage, on='customer_id')\
                 .merge(billing, on='customer_id')\
                 .merge(support, on='customer_id')\
                 .merge(churn, on='customer_id')

# Handling missing values and correcting deprecated method
df.replace("Unresolved", pd.NA, inplace=True)
df.ffill(inplace=True)

# Convert categorical columns to category type
categorical_cols = ['gender', 'services_used', 'payment_method', 'resolutions']
df[categorical_cols] = df[categorical_cols].astype('category')

# Encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Normalize numerical features
numerical_cols = ['age', 'income', 'usage_frequency', 'duration', 'monthly_charges', 'total_charges', 'support_calls', 'issues_raised']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Exploratory Data Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Model Training and Evaluation
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}")

# Feature Importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance')
plt.show()
