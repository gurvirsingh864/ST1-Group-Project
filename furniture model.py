import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Reading the dataset
furniture_data = pd.read_csv('Furniture Price Prediction.csv')

# 2. Problem statement definition
# Predicting the price of furniture based on various features

# 3. Target variable identification
target = 'price'

# 4. Visualizing the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(furniture_data[target], bins=50, kde=True)
plt.title('Distribution of Furniture Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 5. Data exploration at a basic level
print(furniture_data.info())
print(furniture_data.describe())

# 6. Identifying and rejecting useless columns
furniture_data = furniture_data.drop(['url'], axis=1)

# 7. Visual Exploratory Data Analysis (EDA)
for column in furniture_data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(furniture_data[column], bins=50, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 8. Feature selection based on data distribution
# Since correlation-based selection is not working, we use all features
selected_features = [col for col in furniture_data.columns if col != target]
print(f'Using all features: {selected_features}')

# 9. Removal of outliers and missing values
furniture_data = furniture_data[(furniture_data['price'] < furniture_data['price'].quantile(0.99)) & 
                                (furniture_data['price'] > furniture_data['price'].quantile(0.01))]

# 10. Data conversion to numeric values
le = LabelEncoder()
for column in furniture_data.select_dtypes(include=['object']).columns:
    furniture_data[column] = le.fit_transform(furniture_data[column])

# 11. Training/Testing Sampling and K-fold cross validation
X = furniture_data[selected_features]
y = furniture_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 12. Investigating multiple Regression algorithms
models = {
    'Linear Regression': LinearRegression()
}

# 13. Selection of the best model
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
