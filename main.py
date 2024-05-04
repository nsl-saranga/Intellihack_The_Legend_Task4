import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset into a DataFrame
df = pd.read_csv("Watches Bags Accessories.csv")

# Handle missing values (if any)
df = df.dropna(axis='index', how='any')

# Remove "Sold" text and convert "K" notation to actual numbers
df['Sold Count'] = df['Sold Count'].str.replace('Sold', '').str.replace('K', '000').astype(int)

# Normalize numerical features
scaler = MinMaxScaler()
num_features = ['Rating Count', 'Sold Count', 'Current Price', 'Original Price']
df[num_features] = scaler.fit_transform(df[num_features])

# Split the dataset into training and testing sets
X = df.drop(columns=['Sold Count'])  # Features
y = df['Sold Count']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model Training
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

label_encoder = LabelEncoder()
future_data = pd.read_csv("future_data.csv")
future_data.fillna(0, inplace=True)
future_data['Voucher'] = label_encoder.transform(future_data['Voucher'])
future_data['Delivery'] = label_encoder.transform(future_data['Delivery'])
future_data['Category'] = label_encoder.transform(future_data['Category'])
future_data[num_features] = scaler.transform(future_data[num_features])
future_pred = model.predict(future_data)

# Trend nalysis and Interpretation
# # Analyze model predictions and insights to identify significant trends for each product category
#
# # Visualization
# # Visualize forecasted trends and compare them with historical dataA
