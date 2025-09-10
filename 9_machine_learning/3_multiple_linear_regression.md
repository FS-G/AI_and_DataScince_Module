# Advanced Regression Techniques - Complete Lecture Guide

## Dataset Overview: Diabetes Dataset
- **Source**: sklearn.datasets.load_diabetes
- **Features**: 10 physiological variables (age, sex, BMI, blood pressure, etc.)
- **Target**: Quantitative measure of disease progression after one year
- **Samples**: 442 patients
- **Purpose**: Perfect for regression analysis demonstration

---

## 1. Multiple Linear Regression

### What is Multiple Linear Regression?
- **Extension** of simple linear regression with **multiple independent variables**
- **Predicts** one continuous dependent variable using several predictors
- **Assumes linear relationship** between predictors and target

### Mathematical Foundation

#### The Equation
```
y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ + ε
```

Where:
- **y**: Dependent variable (target)
- **β₀**: Intercept (constant term)
- **β₁, β₂, ..., βₙ**: Coefficients (slopes) for each feature
- **x₁, x₂, ..., xₙ**: Independent variables (features)
- **ε**: Error term

#### Matrix Form
```
Y = Xβ + ε
```

#### Normal Equation Solution
```
β = (XᵀX)⁻¹XᵀY
```

### Practical Implementation

#### Using Scikit-Learn
```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

# Load dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Create feature names for better interpretation
feature_names = diabetes.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model coefficients
print("Intercept (β₀):", model.intercept_)
print("Coefficients:")
for i, coef in enumerate(model.coef_):
    print(f"  {feature_names[i]}: {coef:.2f}")
```

#### Using StatsModels (with P-values)
```python
import statsmodels.api as sm

# Add constant term for intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit model
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Print detailed summary
print(model_sm.summary())

# Make predictions
y_pred_sm = model_sm.predict(X_test_sm)
```

### Interpreting Results

#### Coefficients Interpretation
- **Positive coefficient**: As feature increases by 1 unit, target increases by coefficient value
- **Negative coefficient**: As feature increases by 1 unit, target decreases by coefficient value
- **Magnitude**: Larger absolute values indicate stronger influence

#### P-value Interpretation
- **p < 0.001**: Extremely significant (***) 
- **p < 0.01**: Highly significant (**)
- **p < 0.05**: Significant (*)
- **p > 0.05**: Not statistically significant

#### Example Interpretation
```
If BMI coefficient = 938.24 with p-value = 0.000:
"For every 1 unit increase in BMI, disease progression 
increases by 938.24 units (highly significant, p < 0.001)"
```

---

## 2. Regularized Regression

### Why Regularization?
- **Problem**: Overfitting with many features
- **Solution**: Add penalty term to prevent large coefficients
- **Benefits**: Better generalization, feature selection, handles multicollinearity

### Ridge Regression (L2 Regularization)

#### Mathematical Foundation
```
Cost Function = MSE + α × Σ(βᵢ²)
```
```
J(β) = ||y - Xβ||² + α||β||₂²
```

#### Key Characteristics
- **Shrinks coefficients** toward zero but **never exactly zero**
- **Handles multicollinearity** well
- **Good for prediction** when most features are relevant
- **Alpha (α)**: Controls regularization strength

#### Implementation
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Standardize features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_ridge = ridge.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, y_pred_ridge)

print(f"Ridge R² Score: {ridge_r2:.4f}")
print("Ridge Coefficients:")
for i, coef in enumerate(ridge.coef_):
    print(f"  {feature_names[i]}: {coef:.2f}")
```

### Lasso Regression (L1 Regularization)

#### Mathematical Foundation
```
Cost Function = MSE + α × Σ|βᵢ|
```
```
J(β) = ||y - Xβ||² + α||β||₁
```

#### Key Characteristics
- **Shrinks coefficients** and can set them **exactly to zero**
- **Automatic feature selection** (sparse solutions)
- **Good for interpretability** when few features are relevant
- **May struggle** with groups of correlated features

#### Implementation
```python
from sklearn.linear_model import Lasso

# Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_lasso = lasso.predict(X_test_scaled)
lasso_r2 = r2_score(y_test, y_pred_lasso)

print(f"Lasso R² Score: {lasso_r2:.4f}")
print("Lasso Coefficients:")
for i, coef in enumerate(lasso.coef_):
    print(f"  {feature_names[i]}: {coef:.2f}")

# Count non-zero coefficients
non_zero_coefs = np.sum(lasso.coef_ != 0)
print(f"Number of selected features: {non_zero_coefs}")
```

### Alpha Selection
```python
from sklearn.linear_model import RidgeCV, LassoCV

# Cross-validation for optimal alpha
alphas = np.logspace(-3, 2, 50)

# Ridge with CV
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best Ridge Alpha: {ridge_cv.alpha_:.4f}")

# Lasso with CV
lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Best Lasso Alpha: {lasso_cv.alpha_:.4f}")
```

---

## 3. Polynomial Regression

### What is Polynomial Regression?
- **Extension** of linear regression using **polynomial features**
- **Captures non-linear relationships** while remaining linear in parameters
- **Creates new features** by raising existing features to powers

### Mathematical Foundation
#### For one variable:
```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

#### For multiple variables:
```
y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂² + β₅x₁x₂ + ...
```

### Implementation
```python
from sklearn.preprocessing import PolynomialFeatures

# Use first feature (age) for demonstration
X_age = X[:, 0].reshape(-1, 1)  # Age feature only
X_age_train, X_age_test, y_train_age, y_test_age = train_test_split(
    X_age, y, test_size=0.2, random_state=42
)

# Create polynomial features
degrees = [1, 2, 3, 4]
results = {}

for degree in degrees:
    # Generate polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_age_train)
    X_test_poly = poly_features.transform(X_age_test)
    
    # Fit model
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train_age)
    
    # Evaluate
    y_pred_poly = poly_model.predict(X_test_poly)
    r2 = r2_score(y_test_age, y_pred_poly)
    mse = mean_squared_error(y_test_age, y_pred_poly)
    
    results[degree] = {'R²': r2, 'MSE': mse}
    print(f"Degree {degree}: R² = {r2:.4f}, MSE = {mse:.2f}")
```

### Multi-feature Polynomial Regression
```python
# Polynomial features for all variables (degree 2)
poly_all = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_all = poly_all.fit_transform(X_train)
X_test_poly_all = poly_all.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features: {X_train_poly_all.shape[1]}")

# Fit with regularization (recommended for high-degree polynomials)
poly_ridge = Ridge(alpha=1.0)
poly_ridge.fit(X_train_poly_all, y_train)

y_pred_poly_ridge = poly_ridge.predict(X_test_poly_all)
poly_ridge_r2 = r2_score(y_test, y_pred_poly_ridge)
print(f"Polynomial Ridge R²: {poly_ridge_r2:.4f}")
```

---

## 4. Regression Evaluation Metrics

### 1. Mean Squared Error (MSE)
```python
mse = mean_squared_error(y_test, y_pred)
```
- **Formula**: `MSE = (1/n) × Σ(yᵢ - ŷᵢ)²`
- **Range**: [0, ∞)
- **Interpretation**: **Lower is better**, sensitive to outliers
- **Units**: Squared units of target variable

### 2. Root Mean Squared Error (RMSE)
```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```
- **Formula**: `RMSE = √MSE`
- **Range**: [0, ∞)
- **Interpretation**: **Lower is better**, same units as target
- **Use**: Easier to interpret than MSE

### 3. Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
```
- **Formula**: `MAE = (1/n) × Σ|yᵢ - ŷᵢ|`
- **Range**: [0, ∞)
- **Interpretation**: **Lower is better**, robust to outliers
- **Use**: When you want equal weight for all errors

### 4. R² Score (Coefficient of Determination)
```python
r2 = r2_score(y_test, y_pred)
```
- **Formula**: `R² = 1 - (SSres/SStot)`
- **Range**: (-∞, 1]
- **Interpretation**: **Higher is better**
  - **1.0**: Perfect predictions
  - **0.0**: Model performs as well as mean
  - **Negative**: Model performs worse than mean

### 5. Adjusted R²
```python
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

adj_r2 = adjusted_r2(r2, X_test.shape[0], X_test.shape[1])
```
- **Purpose**: Penalizes for additional features
- **Use**: Better for model comparison with different feature counts

### Complete Evaluation Function
```python
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2(r2, X_test.shape[0], X_test.shape[1])
    
    print(f"\n{model_name} Evaluation:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Adj R²: {adj_r2:.4f}")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Adj_R²': adj_r2}

# Evaluate all models
models = {
    'Linear Regression': model,
    'Ridge Regression': ridge,
    'Lasso Regression': lasso
}

results = {}
for name, model in models.items():
    if name == 'Linear Regression':
        results[name] = evaluate_model(model, X_test, y_test, name)
    else:
        results[name] = evaluate_model(model, X_test_scaled, y_test, name)
```

---

## Summary and Best Practices

### When to Use Which Method?
- **Linear Regression**: Simple baseline, interpretable results
- **Ridge Regression**: Many features, multicollinearity, want all features
- **Lasso Regression**: Feature selection needed, sparse solutions desired
- **Polynomial Regression**: Non-linear relationships suspected

### Key Takeaways
- **Always scale features** for regularized regression
- **Use cross-validation** to select hyperparameters
- **Check assumptions**: linearity, independence, homoscedasticity, normality
- **Validate on test set** to assess true performance
- **Consider adjusted R²** when comparing models with different feature counts

### Model Selection Checklist
1. **Start simple**: Linear regression baseline
2. **Check for overfitting**: Compare train vs test performance
3. **Try regularization**: Ridge/Lasso if many features
4. **Consider non-linearity**: Polynomial features if needed
5. **Evaluate thoroughly**: Use multiple metrics
6. **Interpret results**: Understand coefficient meanings and significance