# Housing-Price-Prediction-Machine-Learning-Project-

## Overview

This project demonstrates a complete machine learning workflow for predicting housing prices using a structured dataset.

It includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training and evaluation
* Hyperparameter tuning

The repository is designed as a **learning-focused project**, with both code and detailed explanations of each step.

---

## Project Structure

```
├── housing.csv              # Dataset
├── notebook / script       # Code implementation
├── explanation.md          # Detailed step-by-step explanation
└── README.md               # Project overview
```

---

## What This Project Covers

### 1. Data Preparation

* Handling missing values
* Splitting features and target
* Train-test split

### 2. Exploratory Data Analysis (EDA)

* Histograms to understand distributions
* Correlation heatmaps to analyze relationships
* Geographic visualization of housing prices

### 3. Feature Engineering

* Log transformation to reduce skewness
* One-hot encoding for categorical variables
* Creating new features such as:

  * Bedroom ratio
  * Rooms per household

### 4. Modeling

* Linear Regression (baseline model)
* Random Forest Regressor (non-linear model)

### 5. Model Evaluation

* R² score on test data
* Comparison between models

### 6. Hyperparameter Tuning

* GridSearchCV used to find optimal parameters for Random Forest

---

## Explanation

This repository includes a detailed explanation of the code.

The explanation:

* Breaks down the code step by step
* Explains why each step is performed
* Describes the intuition behind transformations and models
* Interprets visualizations (histograms, heatmaps, scatter plots)

You can find it here:

```
explanation.md
```

---

## Key Learning Points

* Importance of data cleaning and preprocessing
* How feature distributions affect model performance
* Why feature engineering is critical in ML
* Differences between linear and ensemble models
* How hyperparameter tuning improves results

---

## Known Issues / Improvements

* Linear regression is trained on unscaled data but evaluated on scaled data (should be fixed)
* One-hot encoding may cause mismatched columns between train and test sets
* Missing values are dropped instead of being imputed
* No pipeline is used (can improve reproducibility and robustness)

---

## How to Run

This project is designed to be run in Google Colab.

### Steps

1. Open Google Colab:
   [https://colab.research.google.com/](https://colab.research.google.com/)

2. Upload your files:

   * Upload the notebook or script
   * Upload `housing.csv`

3. Install required libraries (if needed):

```bash
!pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the cells step by step

---

### Notes

* Make sure `housing.csv` is in the same directory as your notebook
* If the file is not found, you can upload it manually using:

```python
from google.colab import files
uploaded = files.upload()
```

---


## Goal of This Project

The goal is not just to build a model, but to clearly understand:

* What each step does
* Why it is necessary
* How it affects the final performance

---

## Author

Dea Ceku

---


