
# data=pd.read_csv("C:\\Users\\Roshan\\Downloads\\springs.csv") 


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# Load the dataset
url = "C:\\Users\\Roshan\\Downloads\\springs.csv"
df = pd.read_csv(url)

# Data Preprocessing
# Assuming you have the 'RAINFALL(mm)' and 'SPRING1 DISCHARGE (l/m)' columns in your dataset
X = df[['RAINFALL(mm)']].values
y = df['SPRING1 DISCHARGE (l/m)'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')

# Optionally, you can calculate the accuracy percentage based on R-squared
accuracy_percentage = r_squared * 100
print(f'Accuracy Percentage: {accuracy_percentage}%')

# Visualize the results
plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred, color='blue', label='Predicted values')
plt.xlabel('RAINFALL(mm)')
plt.ylabel('SPRING1 DISCHARGE (l/m)')
plt.legend()
plt.show()

# Use the trained model to make predictions for new data
new_rainfall_values = [[20], [50], [100]]  # Replace with your desired rainfall values
predicted_outflow = model.predict(new_rainfall_values)
print(f'Predicted Outflow for new rainfall values: {predicted_outflow}')

with open('your_model_filename.pkl', 'wb') as file:
    pickle.dump(model, file)
