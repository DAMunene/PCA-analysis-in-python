import pandas as pd
import numpy as np
data = pd.read_csv('C:\\Users\\MOFAT\\Desktop\\Clean.csv')
print(data.shape)
target_variable = data["ID_1"]
selected_columns = ["ID_2","ID_3", "ID_4"]
features_variable = data.loc[:, selected_columns]
print(features_variable.head())
print(data.head(5))
from sklearn.preprocessing import StandardScaler
# Assuming features_variable contains your dataset
# Create an instance of StandardScaler
scaler = StandardScaler()
# Fit the scaler to your data and transform it
standardized_data = scaler.fit_transform(features_variable)

from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca_data = pca.fit_transform(standardized_data)
print(pca_data.shape)
print(pca_data)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.5)
plt.title('PCA Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

import pandas as pd
# Compute the covariance matrix of the standardized data
covariance_matrix = pd.DataFrame(standardized_data, columns=features_variable.columns).cov()
# Display the covariance matrix as a table
print("Covariance Matrix:")
print(covariance_matrix)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_variable, target_variable, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients of the regression model
print("Coefficients:", model.coef_)

# Print the intercept of the regression model
print("Intercept:", model.intercept_)