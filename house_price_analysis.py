import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
csv_path = r'C:\Users\Zenbook\PycharmProjects\pythonProject\zameen-updated.csv'
df = pd.read_csv(csv_path)

# Optimize data types for existing columns
if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
if 'bedrooms' in df.columns:
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
if 'baths' in df.columns:
    df['baths'] = pd.to_numeric(df['baths'], errors='coerce')
if 'Area Size' in df.columns:
    df['Area Size'] = df['Area Size'].replace(r'[^\d.]+', '', regex=True).astype(float)

# Handle missing values separately for numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('price')
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
numeric_imputer = SimpleImputer(strategy='mean')
non_numeric_imputer = SimpleImputer(strategy='most_frequent')

df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])

# Handle infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Feature Engineering
if 'date_added' in df.columns:
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['age'] = 2024 - df['date_added'].dt.year
    df['age'] = df['age'].fillna(df['age'].mean())

df['bedrooms_per_floor'] = df['bedrooms'] / (df['Area Size'] / 272)  # Assuming 1 Marla = 272 sq.ft.

# Function to convert prices to millions and remove decimals
def to_millions(price):
    return np.round(price / 1_000_000, 3)

# 1. Stacked Column Chart of Average Price by Property Type
if 'property_type' in df.columns:
    avg_price_by_property = df.groupby('property_type')['price'].mean().reset_index()
    avg_price_by_property['price'] = avg_price_by_property['price'].apply(to_millions)
    avg_price_by_property = avg_price_by_property.sort_values(by='price', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_price_by_property, x='property_type', y='price', hue='property_type', palette="viridis", dodge=False)
    plt.title('Average Price by Property Type')
    plt.xlabel('Property Type')
    plt.ylabel('Average Price (Millions)')
    plt.xticks(rotation=45)
    plt.legend([], [], frameon=False)
    plt.show()

# 2. Stacked Bar Chart of Maximum (Top 10 Locations by Price)
if 'location' in df.columns:
    max_price_by_location = df.groupby('location')['price'].max().reset_index()
    max_price_by_location['price'] = max_price_by_location['price'].apply(to_millions)
    top_10_locations = max_price_by_location.nlargest(10, 'price')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_10_locations, x='location', y='price', hue='location', palette="viridis", dodge=False)
    plt.title('Maximum Price by Location (Top 10 Locations)')
    plt.xlabel('Location')
    plt.ylabel('Maximum Price (Millions)')
    plt.xticks(rotation=45)
    plt.legend([], [], frameon=False)
    plt.show()

# 3. Stacked Column Chart of Average Price by City
if 'city' in df.columns:
    avg_price_by_city = df.groupby('city')['price'].mean().reset_index()
    avg_price_by_city['price'] = avg_price_by_city['price'].apply(to_millions)
    avg_price_by_city = avg_price_by_city.sort_values(by='price', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_price_by_city, x='city', y='price', hue='city', palette="viridis", dodge=False)
    plt.title('Average Price by City')
    plt.xlabel('City')
    plt.ylabel('Average Price (Millions)')
    plt.xticks(rotation=45)
    plt.legend([], [], frameon=False)
    plt.show()

# Check if 'purpose' column exists
if 'purpose' in df.columns and 'property_type' in df.columns:
    # Create the table with 'Property Type', 'Total For Sale', and 'Total For Rent'
    property_type_count = df.pivot_table(index='property_type', columns='purpose', aggfunc='size', fill_value=0).reset_index()
    property_type_count.columns = ['property_type', 'Total For Rent', 'Total For Sale']
    print("\nTotal Number of Different Property Types for Rent and Sale:")
    print(property_type_count)
else:
    print("Column 'purpose' or 'property_type' does not exist in the dataset.")

# 4. Pie Chart for Average Price by Province
if 'province_name' in df.columns:
    avg_price_by_province = df.groupby('province_name')['price'].mean().reset_index()
    avg_price_by_province['price'] = avg_price_by_province['price'].apply(to_millions)

    plt.figure(figsize=(8, 8))
    plt.pie(avg_price_by_province['price'], labels=avg_price_by_province['province_name'], autopct='%1.1f%%', colors=sns.color_palette("viridis", len(avg_price_by_province['province_name'])))
    plt.title('Average Price by Province')
    plt.show()

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_cols = ['province_name', 'city', 'location', 'property_type', 'purpose']
encoded_categorical_data = encoder.fit_transform(df[categorical_cols])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_cols))
df = df.join(encoded_categorical_df)
df.drop(categorical_cols, axis=1, inplace=True)

# Identify outliers
z_scores = np.abs((df['price'] - df['price'].mean()) / df['price'].std())
outliers = df[z_scores > 3]
print(f'Number of outliers: {len(outliers)}')

print(outliers.describe())

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['price'])
plt.title('Boxplot of House Prices')
plt.xlabel('Price')
plt.show()

# Prepare the data for modeling
# Drop non-relevant columns for prediction
relevant_cols = ['property_id', 'location_id', 'latitude', 'longitude', 'baths', 'bedrooms', 'Area Size', 'age', 'bedrooms_per_floor'] + list(encoded_categorical_df.columns)
X = df[relevant_cols]
y = df['price']

# Ensure there are no missing values in the dataset
if X.isnull().sum().sum() > 0:
    print("There are missing values in the dataset. Imputing again...")
    X.loc[:, numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    X.loc[:, non_numeric_cols] = non_numeric_imputer.fit_transform(X[non_numeric_cols])

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train a random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the models
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Convert MSE and R2 to three decimal places and convert large numbers to millions where appropriate
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f'Linear Regression MSE: {lr_mse:.3f}')
print(f'Linear Regression R2: {lr_r2:.3f}')
print(f'Random Forest MSE: {rf_mse:.3f}')
print(f'Random Forest R2: {rf_r2:.3f}')

# Define the hypothetical scenarios
scenarios = pd.DataFrame({
    'Area Size': [2720, 5440, 1360, 3536],  # Area sizes for 10 Marla, 1 Kanal, 5 Marla, 13 Marla
    'bedrooms': [5, 8, 3, 4],
    'baths': [4, 6, 2, 3],
    'latitude': [33.6844, 24.8607, 31.5497, 32.5734],  # Hypothetical latitudes
    'longitude': [73.0479, 67.0011, 74.3436, 74.3098], # Hypothetical longitudes
    'age': [0, 0, 0, 0],  # Hypothetical age values
    'bedrooms_per_floor': [5 / (2720 / 272), 8 / (5440 / 272), 3 / (1360 / 272), 4 / (3536 / 272)]
})

# Predict the prices for the scenarios
scenario_lr_predictions = lr_model.predict(scenarios)
scenario_rf_predictions = rf_model.predict(scenarios)

# Print the predictions
print("\nPredicted Prices for Scenarios (in Millions):")
print("Linear Regression Predictions:")
for i, pred in enumerate(scenario_lr_predictions):
    print(f"Scenario {i + 1}: ${pred / 1_000_000:.3f} Million")

print("Random Forest Predictions:")
for i, pred in enumerate(scenario_rf_predictions):
    print(f"Scenario {i + 1}: ${pred / 1_000_000:.3f} Million")
