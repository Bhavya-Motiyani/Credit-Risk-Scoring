# Credit Risk Scoring

A machine learning project for **credit risk scoring**, built using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset from Kaggle.

The project focuses not only on predicting credit risk, but also on understanding **why** a customer is classified as high risk using Explainable AI techniques such as **SHAP**.

---

## Project Overview

The original dataset contained approximately **150,000 customer records** and required substantial cleaning and preprocessing before it could be used for modeling.

The project was developed through the following stages:

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Handling Class Imbalance
5. Model Training & Hyperparameter Tuning
6. Model Comparison & Selection
7. SHAP-based Model Explainability
8. Local Flask Web Application

---

## 1. Data Cleaning

The original dataset contained missing values, inconsistent entries, and unrealistic values that required investigation before modeling.

The data cleaning process included:

* Removing unnecessary columns
* Investigating missing `MonthlyIncome` values
* Identifying cases where `DebtRatio` contained suspicious income-like values
* Removing clearly erroneous or unrecoverable records
* Handling missing-value indicators
* Creating a `LogIncome` feature to reduce the effect of income skewness
* Preparing the cleaned dataset for further analysis

The cleaned dataset was saved as:

```text
cleaned_data.csv
```

The final modeling dataset contained **134,206 observations** after the cleaning and preprocessing performed in the notebooks.

---

## 2. Exploratory Data Analysis

After cleaning the dataset, extensive **Exploratory Data Analysis (EDA)** was performed to understand the relationships between customer characteristics and the target variable:

```text
SeriousDlqin2yrs
```

which represents whether the customer experienced serious delinquency within two years.

The analysis explored relationships involving:

* Age
* Income
* Debt Ratio
* Revolving Credit Utilization
* Number of Open Credit Lines
* Number of Real Estate Loans
* Number of Dependents
* 30–59 Days Past Due
* 60–89 Days Past Due
* 90+ Days Late

Additional grouped analyses were performed using variables such as **Age Groups**, delinquency categories, credit-line ranges, and real-estate-loan ranges.

The EDA also resulted in a set of **key groups and customer characteristics to watch out for when assessing credit risk**.

The complete EDA is available in:

```text
EDA.ipynb
```

---

## 3. Feature Engineering & Preprocessing

Several transformations were applied before model training.

These included:

* Standard scaling for selected continuous variables
* Robust scaling for variables affected by outliers
* One-hot encoding of categorical features such as `AgeGroup`
* Log transformation of income
* Creation of missing-value indicators
* Grouping age into meaningful categories

A `ColumnTransformer` was used to keep the preprocessing steps organized and reproducible.

---

## 4. Handling Class Imbalance

Credit default prediction is an imbalanced classification problem, with significantly fewer delinquent customers than non-delinquent customers.

In the cleaned dataset:

| Class          |   Count |
| -------------- | ------: |
| Non-delinquent | 126,189 |
| Delinquent     |   8,017 |

To address this imbalance, different approaches were explored:

* **Class weighting**
* **SMOTENC oversampling**
* **Sample weighting** where appropriate

The goal was to improve the model's ability to identify the minority class rather than simply maximizing overall accuracy.

---

## 5. Models

Three main machine learning algorithms were explored:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

Different approaches to hyperparameter optimization were also explored, including:

* **GridSearchCV**
* **Optuna**
* **Manual hyperparameter tuning**

### Random Forest

For Random Forest, exhaustive GridSearchCV would have been computationally expensive with the available compute, so manual tuning was used instead.

### Logistic Regression

Logistic Regression was tuned using **GridSearchCV**, with recall as the optimization metric.

The best value of `C` found was:

```text
C = 0.01
```

### Gradient Boosting

Gradient Boosting and Random Forest were also explored using **Optuna**.

The Optuna study was configured to maximize **recall**, since identifying delinquent customers is particularly important in a credit-risk setting.

The best Gradient Boosting configuration found was:

```text
n_estimators = 60
learning_rate = 0.010165
min_samples_split = 6
min_samples_leaf = 14
```

---

## 6. Model Comparison

The final models were evaluated on the test set.

| Model                   |   Accuracy | Precision |  Recall | F1 Score |
| ----------------------- | ---------: | --------: | ------: | -------: |
| **Logistic Regression** | **79.40%** |   **18%** | **73%** | **0.29** |
| Gradient Boosting       |     68.26% |       13% | **82%** |     0.23 |

### Logistic Regression

```text
Accuracy:  79.40%
Recall:    73%
Precision: 18%
F1 Score:  0.29
```

Confusion Matrix:

```text
[[25226  6378]
 [  535  1413]]
```

### Gradient Boosting Classifier

```text
Accuracy:  68.26%
Recall:    82%
Precision: 13%
F1 Score:  0.23
```

Confusion Matrix:

```text
[[21311 10293]
 [  355  1593]]
```

Although Gradient Boosting achieved higher recall, it produced substantially more false positives and had lower accuracy, precision, and F1 score.

For this project, **Logistic Regression was selected as the final model** because it provided a better overall balance between identifying risky customers and avoiding excessive false-positive classifications.

---

## 7. Model Explainability with SHAP

The final Logistic Regression model was also analyzed using **SHAP (SHapley Additive exPlanations)**.

A SHAP explainer was used to understand the contribution of individual features to a prediction.

For individual customers, a **SHAP waterfall plot** shows:

* Which features increased the predicted risk
* Which features reduced the predicted risk
* How strongly each feature influenced the prediction

The prediction and SHAP analysis can be found in:

```text
Prediction.ipynb
```

This makes the model more interpretable than simply returning a high-risk or low-risk prediction.

---

## 8. Local Web Application

The trained model was then integrated into a local **Flask web application**.

The application allows a user to enter customer information and receive:

* Predicted credit risk
* Risk probability
* Key factors influencing the prediction

The interface also presents the most important factors contributing to the model's decision.

### Application Preview

![Credit Risk Scoring Application](Screenshot%20%28122%29.png)

The web application was built using:

* **HTML**
* **CSS**
* **JavaScript**
* **Flask**
* **Python**
* **Scikit-learn**

AI assistance was used during the development of the HTML, CSS, and Flask components.

---

## 9. Key Takeaways

* Real-world credit datasets require significant cleaning before modeling.
* Class imbalance makes accuracy alone a poor metric for credit-risk classification.
* **Recall is particularly important** because a false negative can mean incorrectly classifying a risky customer as safe.
* Increasing recall can come at the cost of precision and accuracy.
* Logistic Regression provided the best overall balance among the models evaluated.
* SHAP provides an interpretable view of **why** a particular customer was classified as risky.
* The final model was integrated into a Flask application to demonstrate how the model could be used in a practical setting.

---

## Technologies Used

**Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · SHAP · Optuna · Imbalanced-learn · Flask · HTML · CSS · JavaScript · Jupyter Notebook**

---

## Project Files

| File                                                    | Description                                |
| ------------------------------------------------------- | ------------------------------------------ |
| `Data_cleaning.ipynb`                                   | Data cleaning and preprocessing            |
| `EDA.ipynb`                                             | Exploratory data analysis and key insights |
| `Logistic Regression (best).ipynb`                      | Final Logistic Regression model            |
| `Optuna_tuning_RFC_GB (best_recall_low_accuracy).ipynb` | Random Forest and Gradient Boosting tuning |
| `Prediction.ipynb`                                      | Predictions and SHAP explainability        |
| `cleaned_data.csv`                                      | Cleaned dataset                            |
| `final_logistic_model.pkl`                              | Saved final Logistic Regression model      |
| `GB_model.pkl`                                          | Saved Gradient Boosting model              |
| `app.py`                                                | Flask application                          |


## 👩‍💻 Author

**Bhavya Motiyani**  
B.Tech in Computer Science and Engineering (Data Science Specialization)  
Gujarat Technological University — VGEC  
📧 *bhavyamotiyani68@gmail.com*
🔗 [LinkedIn Profile](https://www.linkedin.com/in/bhavya-motiyani-059544306)
