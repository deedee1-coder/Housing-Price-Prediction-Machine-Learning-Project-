# Housing Price Prediction — Explanation

--- 

## 1. Loading and Inspecting the Data

I start by importing the main libraries I’ll need:

- `pandas` and `numpy` for handling data
- `matplotlib` and `seaborn` for visualization

Then I load the dataset:

```python
data = pd.read_csv("housing.csv")
```

After loading, I check the structure of the dataset:

```python
data.info()
```

![image.png](attachment:0a6fc2e6-57be-4486-8aed-419c28406d61:image.png)

This helps me understand:

- What features exist
- Which columns have missing values
- The data types (numerical vs categorical)

---

## 2. Handling Missing Values

```python
data.dropna(inplace=True)
```

Here, I remove all rows that contain missing values.

I did this because most ML models can't handle missing data directly.

![image.png](attachment:42834cd7-faa2-4f4c-811a-87ea011ab6dd:image.png)

However, this is a simple approach and might remove useful data — in a real project, I would consider imputation instead.

---

## 3. Splitting Features and Target

```python
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
```

- `X` contains all input features
- `y` is the target I want to predict (house prices)

---

## 4. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

I split the data into:

- 80% training data
- 20% testing data

This allows me to train the model on one part and evaluate it on unseen data to check generalization.

---

## 5. Exploratory Data Analysis (EDA)

I recombine the training data for easier analysis:

```python
train_data = X_train.join(y_train)
```

### Histograms

```python
train_data.hist(figsize=(15,8))
```

I use histograms to understand:

- Feature distributions
- Skewness
- Outliers

![image.png](attachment:6a923eed-9751-45b9-9027-388976a6fc9b:image.png)

---

### Correlation Heatmap

```python
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
```

This shows relationships between variables.

I look for:

- Features strongly correlated with `median_house_value`
- Features highly correlated with each other (possible redundancy)

![image.png](attachment:11f840f3-b7b5-4e20-9e5b-f4519698490b:image.png)

---

## 6. Log Transformation

```python
train_data["total_rooms"]= np.log(train_data["total_rooms"] + 1)
```

(I apply this to several columns.)

Why I do this:

- Many features are right-skewed
- Log transformation makes them more normally distributed

This helps models (especially linear ones) perform better.

---

## 7. Handling Categorical Data

```python
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
```

The `ocean_proximity` column is categorical, so I convert it into numerical form using one-hot encoding.

Each category becomes a binary column (0 or 1).

---

## 8. Geographic Visualization

```python
sns.scatterplot(x='latitude', y='longitude', hue='median_house_value')
```

This plot shows:

- House locations
- Color indicates price

I can visually see that:

- Some regions (e.g., near coastlines) have higher prices

![image.png](attachment:39481937-da97-40d4-bbed-8b7eaa3623c0:image.png)

![image.png](attachment:0b0f1968-fdff-488e-9120-7a2f570b00b9:image.png)

---

## 9. Feature Engineering

```python
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']
```

I create new features based on domain intuition.

Why:

- Raw features don’t always capture relationships well
- Ratios often provide better signals

Example:

- `bedroom_ratio` tells how dense bedrooms are in a house

---

## 10. Scaling the Data

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
```

Scaling transforms features so that:

- Mean = 0
- Standard deviation = 1

This is important for models that depend on feature magnitude.

Important observation:

- I scaled the data here, but I didn’t use it for linear regression (this is a mistake).

---

## 11. Linear Regression Model

```python
reg = LinearRegression()
reg.fit(X_train, y_train)
```

I train a linear regression model.

What it does:

- Learns a linear relationship between features and target

It tries to fit:

```
price = w1*x1 + w2*x2 + ... + b
```

---

## 12. Preparing Test Data

I apply the same preprocessing steps to test data:

- Log transformations
- One-hot encoding
- Feature engineering

This is critical — test data must be processed exactly like training data.

---

## 13. Model Evaluation

```python
reg.score(X_test_s, y_test)
```

Issue:

- I trained the model on unscaled data
- But evaluated it on scaled data

This makes the result unreliable.

---

## 14. Random Forest Model

```python
forest = RandomForestRegressor()
forest.fit(X_train_s, y_train)
```

What this model does:

- Builds many decision trees
- Combines their predictions

Why it's useful:

- Captures non-linear relationships
- Handles complex feature interactions

---

```python
forest.score(X_test_s, y_test)
```

This gives the R² score (how well the model explains variance).

---

## 15. Hyperparameter Tuning

```python
param_grid = {
    'n_estimators':[100, 200, 300],
    'min_samples_split': [2, 4, 6, 8],
    'max_depth': [None, 4, 8]
}
```

I define different parameter values to try.

---

```python
grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error")
```

What this does:

- Tries all parameter combinations
- Uses cross-validation (5 splits)
- Selects the best model

---

```python
grid_search.fit(X_train_s, y_train)
```

Runs the search.

---

```python
grid_search.best_estimator_
```

Returns the best model.

---

```python
grid_search.best_estimator_.score(X _test_s, y_test)
```

Error:

- Typo: `X _test_s` should be `X_test_s`

---

## 16. Key Takeaways

- I cleaned the data by removing missing values
- I explored it using histograms and correlation heatmaps
- I transformed skewed features using log scaling
- I converted categorical data using one-hot encoding
- I engineered new features to improve performance
- I trained two models:
    - Linear Regression (baseline)
    - Random Forest (more powerful)
- I improved the model using GridSearchCV

---

## 17. What I Would Fix

- Use scaled data consistently for linear regression
- Ensure train/test columns match after encoding
- Add `random_state` for reproducibility
- Replace `dropna()` with better missing value handling
- Use a Pipeline to avoid preprocessing errors



---
---
---
## Explaining the Graphs
---

## 1. Histograms (Before Log Transformation)

```python
train_data.hist(figsize=(15,8))
```

### What this graph shows

- Each subplot is a histogram of one feature.
- The x-axis = feature values
- The y-axis = frequency (how often values occur)

### What I’m looking for

- Whether the data is **symmetrical** or **skewed**
- Presence of **outliers**
- Whether most values are concentrated in a small range

### What’s happening here

- Features like:
    - `total_rooms`
    - `population`
    - `total_bedrooms`
    
    are **heavily right-skewed** (long tail to the right)
    

### Why this matters

- Many ML models (especially linear regression) assume roughly normal distributions
- Skewed data can:
    - Reduce model performance
    - Make relationships harder to learn

---

## 2. Correlation Heatmap (Before Feature Engineering)

```python
sns.heatmap(train_data.corr(numeric_only=True), annot=True)
```

### What this graph shows

- Pairwise correlation between all numerical features
- Values range from:
    - **+1** → strong positive relationship
    - **0** → no relationship
    - **1** → strong negative relationship

### What I’m focusing on

- Correlation with `median_house_value` (target)
- High correlations between features (multicollinearity)

### What’s happening

- Some features have moderate correlation with price
- Features like `total_rooms`, `total_bedrooms`, and `households` are **highly correlated with each other**

### Why this matters

- High inter-feature correlation can:
    - Confuse linear models
    - Make coefficients unstable

---

## 3. Histograms (After Log Transformation)

```python
train_data.hist(figsize=(15,8))
```

### What changed

- Distributions are now more **balanced and symmetric**
- Long tails are compressed

### Why this is better

- The data now looks closer to a **normal distribution**
- Models can learn relationships more effectively

### Intuition

- Log transformation reduces the impact of very large values

---

## 4. Correlation Heatmap (After Encoding)

```python
sns.heatmap(train_data.corr(numeric_only=True), annot=True)
```

### What’s new

- New columns from one-hot encoding (`ocean_proximity`)

### What I’m checking

- Whether location-related categories correlate with price

### Insight

- Some location categories likely show stronger correlation with house value

---

## 5. Geographic Scatter Plot

```python
sns.scatterplot(x='latitude', y='longitude', hue='median_house_value')
```

### What this graph shows

- Each point = a house
- Position = geographic location
- Color = house price

### What I’m noticing

- Clusters of similar colors (prices)
- Expensive houses are not randomly distributed

### Key insight

- Location is **very important**
- For example:
    - Coastal areas tend to have higher prices

### Why this matters

- It suggests:
    - The model should capture spatial relationships
    - Features like latitude/longitude are valuable

---

## 6. Correlation Heatmap (After Feature Engineering)

```python
sns.heatmap(train_data.corr(numeric_only=True), annot=True)
```

### What’s new

- Added features:
    - `bedroom_ratio`
    - `household_rooms`

### What I’m checking

- Do these new features correlate better with the target?

### What’s happening

- These engineered features often:
    - Show stronger correlation with `median_house_value`
    - Capture relationships that raw features missed

### Why this matters

- This confirms that **feature engineering improved the dataset**
