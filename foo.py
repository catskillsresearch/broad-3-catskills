import numpy as np
from cuml.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
n_samples = 100  # Number of samples
n_features = 3   # Number of input features
n_outputs = 3    # Number of output targets

# Inputs (X) and outputs (y)
X = np.random.rand(n_samples, n_features).astype(np.float32)  # Shape: (100, 3)
y = np.random.rand(n_samples, n_outputs).astype(np.float32)   # Shape: (100, 3)

# Split into training and testing sets
split_ratio = 0.8
split_index = int(n_samples * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize cuML Ridge regression model
alpha = 1.0  # Regularization strength
ridge_model = Ridge(alpha=alpha, fit_intercept=True, solver="eig")

# Train the model
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Print predictions for verification
print("\nSample Predictions:")
print("True values:\n", y_test[:5])
print("Predicted values:\n", y_pred[:5])
