# Import required libraries
from cuml import Ridge
from cuml.model_selection import train_test_split
from cuml.metrics.regression import r2_score
import numpy as np

# Generate synthetic data
n_samples = 1000
n_features = 10
n_outputs = 3  # Multiple outputs

# Create random data (CUDA requires float32)
X = np.random.rand(n_samples, n_features).astype(np.float32)
y = np.random.rand(n_samples, n_outputs).astype(np.float32)  # 2D target array

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Ridge regression model
model = Ridge(
    alpha=1.0,  # Regularization strength
    solver='eig',  # Eigen decomposition solver (good for multi-output)
    fit_intercept=True,
    handle=np.nan  # How to handle missing values
)

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print R² scores for each output
for i in range(n_outputs):
    score = r2_score(y_test[:, i], y_pred[:, i])
    print(f"Output {i+1} R²: {score:.4f}")
