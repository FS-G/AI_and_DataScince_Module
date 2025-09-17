

## What is Ensembling?

**Problem**: Single model can make mistakes

**Solution**: Combine multiple models!

**Think of it like:**
- Asking multiple experts for advice
- Taking their average opinion
- Usually better than any single expert

**Types of Ensembles:**
- **Bagging**: Train models in parallel
- **Boosting**: Train models sequentially 
- **Stacking**: Use one model to combine others

## Bagging (Bootstrap Aggregating)

**Concept**: 
- Create multiple datasets by sampling with replacement
- Train separate models on each dataset
- Combine predictions by voting/averaging

**Bootstrap Sampling Example:**
```
Original: [1, 2, 3, 4, 5]

Sample 1: [1, 1, 3, 4, 5] (1 appears twice)
Sample 2: [2, 3, 3, 4, 4] (some numbers repeat)
Sample 3: [1, 2, 2, 5, 5] (some numbers missing)
```

## Bagging Step-by-Step

**Our Original Dataset:**
| Row | Feature A | Feature B | Target |
|-----|-----------|-----------|---------|
| 1   | 2         | 3         | 1       |
| 2   | 5         | 4         | 0       |
| 3   | 3         | 7         | 1       |
| 4   | 6         | 8         | 0       |
| 5   | 9         | 1         | 1       |

**Step 1**: Create bootstrap samples (sample with replacement)

**Bootstrap Sample 1**: Rows [2, 3, 2, 4, 5]
**Bootstrap Sample 2**: Rows [1, 5, 4, 1, 3] 
**Bootstrap Sample 3**: Rows [5, 3, 4, 5, 2]

**Step 2**: Train one model on each sample

**Step 3**: For prediction, combine all models:
- **Classification**: Majority vote
- **Regression**: Average

## Bagging Code Example

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Create sample data
X = np.array([[2, 3], [5, 4], [3, 7], [6, 8], [9, 1]])
y = np.array([1, 0, 1, 0, 1])

# Bagging with decision trees
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,  # 10 different models
    random_state=42
)

# Train the ensemble
bagging.fit(X, y)

# Make predictions
predictions = bagging.predict(X)
print("Bagging Predictions:", predictions)

# Get prediction probabilities
probabilities = bagging.predict_proba(X)
print("Prediction Probabilities:", probabilities)
```

## Boosting - Learning from Mistakes

**Concept**: Train models one after another
- Each new model focuses on previous mistakes
- Give more weight to wrongly predicted samples
- Final prediction = weighted combination

**Process:**
1. Train Model 1 on original data
2. Find samples Model 1 got wrong
3. Train Model 2, focusing more on those mistakes
4. Repeat...
5. Combine all models with weights

## Boosting Example

**Round 1:**
- Model 1 correctly predicts: [✓, ✗, ✓, ✓, ✗]
- Mistakes on samples 2 and 5

**Round 2:**
- Give higher weight to samples 2 and 5
- Model 2 focuses more on these difficult cases
- Model 2 predictions: [✗, ✓, ✗, ✓, ✓]

**Final Prediction:**
- Combine Model 1 + Model 2 with weights
- Usually more accurate than either alone

## Boosting Code Example

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# AdaBoost Example
ada_boost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_boost.fit(X, y)
ada_predictions = ada_boost.predict(X)

# Gradient Boosting Example  
gb_boost = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_boost.fit(X, y)
gb_predictions = gb_boost.predict(X)

print("AdaBoost Predictions:", ada_predictions)
print("Gradient Boosting Predictions:", gb_predictions)
```







## Random Forest - The Star!

**What is Random Forest?**
- Special type of **Bagging** using **Decision Trees**
- Adds extra randomness for better results

**Two Types of Randomness:**
1. **Random Sampling**: Each tree trained on different bootstrap sample
2. **Random Features**: Each split considers only random subset of features

**Why "Forest"?**
- Many trees = Forest
- Each tree votes on final prediction

## Random Forest Process

**Step 1**: Create multiple bootstrap samples from training data

**Step 2**: For each sample, grow a decision tree with twist:
- At each split, randomly select subset of features
- Choose best split only from these random features
- Not all features considered at each split

**Step 3**: Train many trees (typically 100-1000)

**Step 4**: For prediction:
- **Classification**: Majority vote from all trees
- **Regression**: Average of all tree predictions

## Random Forest Benefits

**Advantages over Single Decision Tree:**

**Reduced Overfitting**: 
- Individual trees might overfit
- Averaging reduces this problem

**Better Generalization**:
- Works well on unseen data
- More robust predictions

**Feature Importance**:
- Shows which features are most important
- Helps understand your data

**Handles Missing Values**:
- Can work with incomplete data

**Works with Large Datasets**:
- Efficient and scalable

## Random Forest Code Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, 
                          n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Random Forest
rf = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Maximum depth of each tree
    min_samples_split=5, # Minimum samples to split
    random_state=42
)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Random Forest Accuracy: {accuracy:.3f}")
```

## Feature Importance with Random Forest

```python
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importance
importance = rf.feature_importances_
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest - Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Top 5 Most Important Features:")
print(importance_df.head())
```

## Comparing All Methods

```python
from sklearn.metrics import accuracy_score
import pandas as pd

# Compare different methods
models = {
    'Single Decision Tree': DecisionTreeClassifier(random_state=42),
    'Bagging': BaggingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

# Display results
results_df = pd.DataFrame(list(results.items()), 
                         columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False)

print("Model Comparison:")
print(results_df)
```

## When to Use Each Method?

**Single Decision Tree:**
- Small datasets
- Need interpretable model
- Fast predictions required

**Bagging/Random Forest:**
- Reduce overfitting
- Better accuracy needed
- Have enough computational resources
- Don't need perfect interpretability

**Boosting (AdaBoost/Gradient Boosting):**
- Want highest possible accuracy
- Sequential learning beneficial
- Willing to tune hyperparameters carefully

**Random Forest is often the best starting point!**

## Hyperparameter Tuning for Random Forest

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit grid search
rf_grid.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", rf_grid.best_params_)
print("Best Cross-Validation Score:", rf_grid.best_score_)

# Test with best model
best_rf = rf_grid.best_estimator_
test_accuracy = accuracy_score(y_test, best_rf.predict(X_test))
print("Test Accuracy with Best Model:", test_accuracy)
```

## Visualizing Random Forest Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract one tree from the forest for visualization
single_tree = rf.estimators_[0]  # First tree

plt.figure(figsize=(15, 10))
plot_tree(single_tree, 
          max_depth=3,  # Show only top 3 levels
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("One Tree from Random Forest (First 3 levels)")
plt.show()

print(f"This tree has {single_tree.tree_.node_count} nodes")
print(f"This tree has depth {single_tree.tree_.max_depth}")
```