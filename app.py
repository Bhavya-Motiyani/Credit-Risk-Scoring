from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import shap

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

with open("final_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

preprocessor = model.named_steps["Preprocessing"]
logistic_model = model.named_steps["Logistic Model"]


# ============================================================
# LOAD BACKGROUND DATA FOR SHAP
# ============================================================

df = pd.read_csv("cleaned_data.csv")

# Feature engineering
df["LogIncome"] = np.log(df["MonthlyIncome"])

bins = [19, 35, 50, 65, float("inf")]
labels = ["Young", "Middle-aged", "Senior", "Old"]

df["AgeGroup"] = pd.cut(
    df["age"],
    bins=bins,
    labels=labels
)


# These are the exact features used by your model
features = [
    "age",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "DebtRatio",
    "DebtRatioMissing",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "LogIncome",
    "AgeGroup"
]

X_background = df[features].copy()

X_background_transformed = preprocessor.transform(
    X_background
)


# ============================================================
# SHAP FEATURE NAMES
# ============================================================

# Remove:
# StandardScaling__
# RobustScaling__
# OHE__
# remainder__

feature_names = np.array([
    name.split("__", 1)[-1]
    for name in preprocessor.get_feature_names_out()
])


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.Explainer(
    logistic_model,
    X_background_transformed,
    feature_names=feature_names
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_sample(form):

    age = int(form["age"])
    monthly_income = float(form["monthly_income"])
    debt_ratio = float(form["debt_ratio"])

    sample = pd.DataFrame([{

        "age": age,

        "RevolvingUtilizationOfUnsecuredLines":
            float(form["utilization"]),

        "NumberOfTime30-59DaysPastDueNotWorse":
            int(form["late_30_59"]),

        "NumberOfTime60-89DaysPastDueNotWorse":
            int(form["late_60_89"]),

        "NumberOfTimes90DaysLate":
            int(form["late_90"]),

        "DebtRatio":
            debt_ratio,

        "DebtRatioMissing":
            0,

        "NumberOfOpenCreditLinesAndLoans":
            int(form["open_credit_lines"]),

        "NumberRealEstateLoansOrLines":
            int(form["real_estate_loans"]),

        "NumberOfDependents":
            int(form["dependents"]),

        # ENGINEERED FEATURE 1
        "LogIncome":
            np.log(monthly_income),

        # ENGINEERED FEATURE 2
        "AgeGroup":
            pd.cut(
                [age],
                bins=bins,
                labels=labels
            )[0]
    }])

    return sample


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability <= 0.30:
        return "Low Risk", "low"

    elif probability <= 0.60:
        return "Medium Risk", "medium"

    else:
        return "High Risk", "high"


# ============================================================
# FLASK ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        try:

            # --------------------------------------------
            # 1. CREATE SAMPLE + ENGINEER FEATURES
            # --------------------------------------------

            sample = create_sample(request.form)


            # --------------------------------------------
            # 2. PREDICT RISK PROBABILITY
            # --------------------------------------------

            probability = model.predict_proba(
                sample
            )[0][1]


            # --------------------------------------------
            # 3. CLASSIFY RISK
            # --------------------------------------------

            risk, risk_class = classify_risk(
                probability
            )


            # --------------------------------------------
            # 4. TRANSFORM DATA
            # --------------------------------------------

            sample_transformed = preprocessor.transform(
                sample
            )


            # --------------------------------------------
            # 5. SHAP VALUES
            # --------------------------------------------

            shap_values = explainer(
                sample_transformed
            )

            values = shap_values.values[0]


            # --------------------------------------------
            # 6. TOP 4 FEATURES
            # --------------------------------------------

            top_indices = np.argsort(
                np.abs(values)
            )[::-1][:4]


            key_factors = []

            for index in top_indices:

                feature = feature_names[index]

                contribution = values[index]

                if contribution > 0:
                    direction = "Increases risk"
                else:
                    direction = "Reduces risk"

                key_factors.append({
                    "name": feature,
                    "direction": direction
                })


            # --------------------------------------------
            # RESULT
            # --------------------------------------------

            result = {
                "risk": risk,
                "risk_class": risk_class,
                "probability": round(
                    probability * 100,
                    2
                ),
                "key_factors": key_factors
            }


        except Exception as e:

            result = {
                "error": str(e)
            }


    return render_template(
        "index.html",
        result=result,
        form_data=request.form
    )


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )