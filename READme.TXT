# Titanic Survival Prediction 🛳️

Predicting survival on the Titanic using passenger attributes with a focus on feature engineering, missing value strategies, and model explainability.

---

## **Project Goal**
The objective of this project is to predict whether a passenger survived the Titanic disaster. The project emphasizes:
- **Feature creation**: Titles from names, family size, cabin presence.
- **Missing value handling**: Strategies for age, cabin, and other incomplete data.
- **Model explainability**: Using SHAP values and feature importance to understand model decisions.

---

## **Requirements**
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`

---

## **Feature Engineering**
- Extract **titles** from passenger names (Mr, Mrs, Miss, etc.)
- Calculate **family size**: `SibSp + Parch + 1`
- Create **cabin presence**: whether a passenger has a cabin number

---

## **Handling Missing Data**
- **Age**: Imputed using median or predictive models
- **Cabin**: Converted to a binary feature (presence/absence)
- Encode categorical features (Sex, Embarked, Title) for ML models

---

## **Model Training**
- Train classification models (Random Forest, XGBoost, Logistic Regression)
- Evaluate using accuracy, F1-score, and ROC-AUC
- Interpret predictions with **SHAP values** for feature importance

---

## **Results**
- Example of model metrics:
  - Accuracy: 0.82
  - F1-Score: 0.81
  - ROC-AUC: 0.87
- Top features influencing survival:
  - Title
  - Sex
  - Age
  - Family size
  - Cabin presence

---

## **Inference Example**
```python
import pandas as pd
from model import predict_survival

# Example passenger
passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex': ['male'],
    'Age': [22],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': ['S']
})

prediction = predict_survival(passenger)
print(f"Predicted Survival: {prediction}")