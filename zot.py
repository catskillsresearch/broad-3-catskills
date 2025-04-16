import cuml
import cupy as cp
from cuml.linear_model import Ridge
from cuml.datasets import make_regression
from sklearn.metrics import r2_score

# Create synthetic multi-output regression data
# 1000 samples, 20 features, and 3 targets
X_cu, y_cu = make_regression(n_samples=1000, n_features=20, n_informative=15,
                             n_targets=3, noise=0.1, random_state=42)

# X_cu and y_cu are CuPy arrays (GPU memory)
# Split into train and test sets
from cuml.preprocessing.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cu, y_cu, test_size=0.2, random_state=42)

# Create and fit Ridge regression model
ridge_model = Ridge(alpha=1.0, fit_intercept=True)
ridge_model.fit(X_train, y_train)

# Predict on test set
y_pred = ridge_model.predict(X_test)

# Evaluate performance (convert to NumPy for sklearn's r2_score)
r2 = r2_score(cp.asnumpy(y_test), cp.asnumpy(y_pred))
print(f"R² score (multi-output Ridge): {r2:.4f}")
