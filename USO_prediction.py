import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Reading the dataset
uso_data = pd.read_csv('FINAL_USO.csv')

# 2. Problem statement definition
# Predicting the closing price of USO based on various features

# 3. Target variable identification
target = 'Close'

# 4. Visualizing the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(uso_data[target], bins=50, kde=True)
plt.title('Distribution of USO Closing Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()

# 5. Data exploration at a basic level
print(uso_data.info())
print(uso_data.describe())

# 6. Identifying and rejecting useless columns
uso_data = uso_data.drop(['Date'], axis=1)

# 7. Visual Exploratory Data Analysis (EDA)
numeric_columns = uso_data.select_dtypes(include=['float64', 'int64']).columns
num_cols = len(numeric_columns)

# Displaying histograms in groups of 8
cols_per_row = 4
rows_per_fig = 2
for i in range(0, num_cols, cols_per_row * rows_per_fig):
    fig, axes = plt.subplots(rows_per_fig, cols_per_row, figsize=(16, 10))
    for j, ax in enumerate(axes.flatten()):
        if i + j < num_cols:
            sns.histplot(uso_data[numeric_columns[i + j]], bins=50, kde=True, ax=ax)
            ax.set_title(f'Distribution of {numeric_columns[i + j]}')
            ax.set_xlabel(numeric_columns[i + j])
            ax.set_ylabel('Frequency')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# 8. Feature selection based on data distribution
corr = uso_data.corr(numeric_only=True)
plt.figure(figsize=(20, 15))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 9. Removal of outliers and missing values
uso_data = uso_data[(uso_data['Close'] < uso_data['Close'].quantile(0.99)) & 
                    (uso_data['Close'] > uso_data['Close'].quantile(0.01))]

# 10. Visual and Statistical Correlation analysis
selected_features = corr[target][corr[target].abs() > 0.3].index.tolist()
if target in selected_features:
    selected_features.remove(target)

print(f'Selected Features: {selected_features}')

# 11. Data conversion to numeric values
# USO dataset doesn't have categorical variables, so no Label Encoding is needed here.

# 12. Training/Testing Sampling and K-fold cross validation
X = uso_data[selected_features]
y = uso_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 13. Investigating multiple Regression algorithms
models = {
    'Linear Regression': LinearRegression()
}

# 14. Selection of the best model
best_model = None
best_score = float('inf')

for name, model in models.items():
    try:
        score = -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf).mean()
        if score < best_score:
            best_score = score
            best_model = model
        print(f'{name}: {score}')
    except Exception as e:
        print(f"Error with {name}: {e}")

if best_model is not None:
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f'Best Model: {best_model.__class__.__name__}')
    print(f'Mean Squared Error on Test Set: {mean_squared_error(y_test, y_pred)}')
else:
    print("No valid model was selected.")

