import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline

# Assume df is the preprocessed dataframe with features and target variable
df = pd.read_csv("output_file.csv")

# Drop unnecessary columns
df = df.drop(["DISTRICT", "Date"], axis=1)

# Handle any missing values (if any)
df = df.dropna()

# Split the data into features (X) and target variable (y)
X = df[["Month_sin", "Month_cos"]]
y = df["Rainfall"]

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb_model', XGBRegressor())
])

# Standardize the features
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Initialize the XGBoost model
# model = XGBRegressor()

# Train the model
# model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')

# Optionally, you can calculate the accuracy percentage based on R-squared
accuracy_percentage = r_squared * 100
print(f'Accuracy Percentage: {accuracy_percentage}%')

# Visualize the results
plt.plot(y_test.index, y_test, label='Actual Rainfall', marker='o', linestyle='-', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Rainfall', marker='o', linestyle='-', color='orange')

plt.title('Actual vs. Predicted Rainfall')
plt.xlabel('Index or Time')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

with open('rainfall.pkl', 'wb') as file:
    pickle.dump(model_pipeline, file)


