# Wine & MNIST — Machine Learning Project

Applied machine learning project with two parts:
- **Regression:** Predicting red wine quality from physicochemical features  
- **Classification:** Distinguishing even vs odd digits in the MNIST dataset  

The project demonstrates the full ML workflow: data exploration, visualization, model building, evaluation, cross-validation, and result reporting.

---

## Problem 1 — Wine Quality (Regression)

**Goal:** Predict wine quality scores from chemical properties.  
**Dataset:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/static/public/186/wine+quality.zip)

**Key steps:**
- Exploratory Data Analysis (descriptive statistics, histograms, correlations)  
- Stratified 80/20 train–test split  
- StandardScaler + Linear Regression  
- Metrics: R², MAE, MAPE, MSE, Accuracy within ±0.5 tolerance  
- Actual vs Predicted plot  
- 10-fold Cross Validation with test R² comparison  

**Outputs:** metrics JSON + plots (histograms, correlation heatmap, actual vs predicted)

---

## Problem 2 — MNIST (Even vs Odd Classification)

**Goal:** Binary classification of digits as even (0) or odd (1).  
**Dataset:** [MNIST on OpenML](https://www.openml.org/d/554)

**Key steps:**
- 85/15 train–test split with stratification  
- Visualization of sample images  
- Logistic Regression pipeline with StandardScaler  
- 3-fold Cross Validation (Accuracy, Recall, Precision)  
- Comparison with a Dummy baseline classifier  
- Confusion matrices for training and test sets  
- Visualization of false positive and false negative examples  

**Outputs:** metrics JSON + plots (sample digits, confusion matrices, FP/FN examples)

---

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
python wa1_wine_and_mnist.py


