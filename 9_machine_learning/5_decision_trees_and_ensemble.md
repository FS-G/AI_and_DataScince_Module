# Decision Trees and Random Forest

## What is a Decision Tree?

A decision tree is like asking a series of **yes/no questions** to make a decision.

**Think of it like this:**
- You want to decide if you should go outside
- Question 1: Is it raining? 
  - If YES → Stay inside
  - If NO → Ask next question
- Question 2: Is it sunny?
  - If YES → Go outside
  - If NO → Maybe stay inside

**In Machine Learning:**
- Each question = **Internal Node**
- Each answer path = **Branch** 
- Final decision = **Leaf Node**

## Decision Tree Structure

```
Root Node (Income > $50k?)
├── YES: Go to next question (Age > 30?)
│   ├── YES: Buy House (LEAF)
│   └── NO: Don't Buy (LEAF)
└── NO: Don't Buy House (LEAF)
```

**Components:**
- **Root Node**: Starting point with best splitting feature
- **Internal Nodes**: Questions that split the data
- **Branches**: Possible answers (YES/NO for binary splits)
- **Leaf Nodes**: Final predictions

**Tree Structure Visualization:**
```
                Root Node
              (Income > $50k?)
                   |
          ┌────────┴────────┐
          │                 │
        YES                NO
          │                 │
    Internal Node      Leaf Node
    (Age > 30?)       (Don't Buy)
          |
     ┌────┴────┐
     │         │
   YES        NO
     │         │
 Leaf Node  Leaf Node
(Buy House) (Don't Buy)
```

## How Does a Decision Tree Learn?

**Step 1:** Start with all data at root

**Step 2:** Find the **best question** to split data
- Which feature separates classes best?
- Use metrics like Gini Impurity or Entropy

**Step 3:** Split data based on that question

**Step 4:** Repeat for each subset until:
- All data points have same label
- No more useful splits possible
- Maximum depth reached

## Simple Example - Will You Play Tennis?

**Dataset:**
| Weather | Temperature | Play Tennis |
|---------|-------------|-------------|
| Sunny   | Hot         | No          |
| Sunny   | Cool        | Yes         |
| Rainy   | Cool        | Yes         |
| Rainy   | Hot         | No          |

**Decision Tree:**
```
Weather?
├── Sunny → Temperature?
│   ├── Hot → No
│   └── Cool → Yes  
└── Rainy → Temperature?
    ├── Hot → No
    └── Cool → Yes
```

## Splitting Criteria - How to Choose Best Split?

**Gini Impurity**: Measures how "impure" a split is
- 0 = Pure (all same class)
- 0.5 = Maximum impurity (50-50 split)

**Entropy**: Measures information gain
- 0 = Pure 
- 1 = Maximum uncertainty

**Goal**: Choose split that **reduces impurity the most**

## Decision Tree Code Example

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample data
data = {
    'Income': [50000, 30000, 70000, 45000, 80000],
    'Age': [25, 35, 45, 28, 50],
    'Children': [0, 2, 1, 0, 3],
    'Buy_House': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['Income', 'Age', 'Children']]
y = df['Buy_House']

# Create and train decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Make predictions
predictions = dt.predict(X)
print("Predictions:", predictions)
```

## Visualizing Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt, 
          feature_names=['Income', 'Age', 'Children'],
          class_names=['No', 'Yes'], 
          filled=True, 
          rounded=True)
plt.title("Decision Tree: Will Buy House?")
plt.show()
```

**What you'll see:**
- Boxes showing the splitting condition
- Colors indicating the majority class
- Numbers showing samples in each node

## The Problem - Overfitting!

**Decision trees can become too complex:**
- They memorize training data
- Poor performance on new data
- Very deep trees with many branches

**Example of Overfitting:**
```
Income > $50k?
├── YES → Age > 30?
│   ├── YES → Children > 2?
│   │   ├── YES → Day = Monday? (TOO SPECIFIC!)
│   │   └── NO → Buy House
│   └── NO → Don't Buy
└── NO → Don't Buy
```

## Preventing Overfitting

**Techniques:**

**Max Depth**: Limit tree height
- `max_depth=5`

**Min Samples Split**: Minimum samples needed to split
- `min_samples_split=10`

**Min Samples Leaf**: Minimum samples in leaf node
- `min_samples_leaf=5`

```python
# Better decision tree with constraints
dt_improved = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```


## Complete Example - Building Decision Tree with Gini

Let's work through a complete example using the "Cool As Ice" dataset:

**Original Dataset:**
| Loves Popcorn | Loves Soda | Age | Cool As Ice |
|---------------|------------|-----|-------------|
| Yes           | Yes        | 7   | No          |
| Yes           | No         | 12  | No          |
| No            | Yes        | 18  | Yes         |
| No            | Yes        | 35  | Yes         |
| Yes           | Yes        | 38  | Yes         |
| Yes           | No         | 50  | No          |
| No            | No         | 83  | No          |

**Target Distribution:**
- Cool As Ice = Yes: 3 samples
- Cool As Ice = No: 4 samples
- Total: 7 samples

## Step 1: Calculate Root Gini Impurity

**Root Gini Impurity:**
```
Gini = 1 - (P_yes)² - (P_no)²
Gini = 1 - (3/7)² - (4/7)²
Gini = 1 - (0.429)² - (0.571)²
Gini = 1 - 0.184 - 0.327
Gini = 0.489
```

## Step 2: Calculate Gini for Each Feature Split

**Feature 1: Loves Popcorn**
- **Loves Popcorn = Yes**: [No, No, Yes, No] → 1 Yes, 3 No
  - Gini = 1 - (1/4)² - (3/4)² = 1 - 0.063 - 0.563 = 0.375
- **Loves Popcorn = No**: [Yes, Yes, No] → 2 Yes, 1 No  
  - Gini = 1 - (2/3)² - (1/3)² = 1 - 0.444 - 0.111 = 0.444

**Weighted Gini for Loves Popcorn:**
```
Weighted_Gini = (4/7) × 0.375 + (3/7) × 0.444 = 0.405
```

**Feature 2: Loves Soda**
- **Loves Soda = Yes**: [No, Yes, Yes, Yes] → 3 Yes, 1 No
  - Gini = 1 - (3/4)² - (1/4)² = 1 - 0.563 - 0.063 = 0.375
- **Loves Soda = No**: [No, No, No] → 0 Yes, 3 No
  - Gini = 1 - (0/3)² - (3/3)² = 1 - 0 - 1 = 0.0

**Weighted Gini for Loves Soda:**
```
Weighted_Gini = (4/7) × 0.375 + (3/7) × 0.0 = 0.214
```

**Feature 3: Age (let's try Age < 15)**
- **Age < 15**: [No, No] → 0 Yes, 2 No
  - Gini = 1 - (0/2)² - (2/2)² = 0.0
- **Age ≥ 15**: [Yes, Yes, Yes, No, No] → 3 Yes, 2 No
  - Gini = 1 - (3/5)² - (2/5)² = 1 - 0.36 - 0.16 = 0.48

**Weighted Gini for Age < 15:**
```
Weighted_Gini = (2/7) × 0.0 + (5/7) × 0.48 = 0.343
```

## Step 3: Choose Best Split

**Information Gain = Root Gini - Weighted Gini**

- **Loves Popcorn**: 0.489 - 0.405 = 0.084
- **Loves Soda**: 0.489 - 0.214 = 0.275 ← **HIGHEST**
- **Age < 15**: 0.489 - 0.343 = 0.146

**Winner**: Loves Soda (lowest Gini = 0.214, highest information gain = 0.275)

## Step 4: Build the Complete Tree

```
Root: Loves Soda?
├── TRUE (Loves Soda = Yes): [7, 12, 18, 35, 38] → [No, No, Yes, Yes, Yes]
│   │   Need further split: Age < 12.5?
│   ├── TRUE (Age < 12.5): [7, 12] → [No, No] 
│   │   └── LEAF: Cool As Ice = No
│   └── FALSE (Age ≥ 12.5): [18, 35, 38] → [Yes, Yes, Yes]
│       └── LEAF: Cool As Ice = Yes
│
└── FALSE (Loves Soda = No): [12, 50, 83] → [No, No, No]
    └── LEAF: Cool As Ice = No
```

## Step 5: Code Implementation

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create the dataset
data = {
    'Loves_Popcorn': [1, 1, 0, 0, 1, 1, 0],
    'Loves_Soda': [1, 0, 1, 1, 1, 0, 0], 
    'Age': [7, 12, 18, 35, 38, 50, 83],
    'Cool_As_Ice': [0, 0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df[['Loves_Popcorn', 'Loves_Soda', 'Age']]
y = df['Cool_As_Ice']

print("Dataset:")
print(df)

# Create decision tree with Gini criterion
dt = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=3,
    min_samples_leaf=1,
    random_state=42
)

dt.fit(X, y)

# Plot the tree
plt.figure(figsize=(15, 10))
plot_tree(dt, 
          feature_names=['Loves_Popcorn', 'Loves_Soda', 'Age'],
          class_names=['Not Cool', 'Cool As Ice'], 
          filled=True, 
          rounded=True,
          fontsize=12)
plt.title("Decision Tree: Cool As Ice Prediction")
plt.show()

# Make predictions
predictions = dt.predict(X)
print("\nPredictions vs Actual:")
comparison = pd.DataFrame({
    'Actual': y,
    'Predicted': predictions,
    'Correct': y == predictions
})
print(comparison)
```

## Step 6: Understanding the Results

**Final Tree Interpretation:**
1. **First Question**: Do you love soda?
   - If **NO** → You're **NOT Cool As Ice**
   - If **YES** → Go to next question

2. **Second Question**: Are you younger than 12.5?
   - If **YES** → You're **NOT Cool As Ice** 
   - If **NO** → You're **Cool As Ice**

**Key Insights:**
- Loving soda is the most important factor
- Among soda lovers, age matters (older = cooler!)
- The tree perfectly classifies this small dataset
- But with only 1 sample in some leaves, it might overfit

## Why This Tree Might Overfit

**Problem with Small Leaf Nodes:**
- Some leaves have very few samples (like Age < 12.5 with only 1 sample)
- Hard to have confidence in predictions for new data
- Tree memorized specific cases rather than learning patterns

**Solutions:**
- Use `min_samples_leaf=2` or higher
- Collect more data
- Use ensemble methods like Random Forest


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




