"""Generate a SYNTHETIC Telco-style dataset for testing the app.

This is NOT the real Telco Customer Churn dataset. It mimics its columns and
contains a few deliberate data-quality problems (blank TotalCharges, duplicate
rows) so you can see the cleaning module working.

Usage:  python generate_sample_data.py   ->  data/sample_telco_churn.csv
"""
import numpy as np
import pandas as pd


def make_sample(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    bonus = np.select([contract == "One year", contract == "Two year"], [8, 18], 0)
    tenure = np.clip((rng.gamma(2.0, 12.0, n) + bonus).astype(int), 0, 72)
    tenure[rng.choice(n, 10, replace=False)] = 0

    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n, p=[0.42, 0.58]))
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def addon(p_yes):
        return np.where(internet == "No", "No internet service",
                        rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]))

    security, backup, protection = addon(0.30), addon(0.35), addon(0.35)
    tech, tv, movies = addon(0.30), addon(0.40), addon(0.40)
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.33, 0.22, 0.22, 0.23])

    monthly = (18 + np.where(phone == "Yes", 4, 0)
               + np.select([internet == "DSL", internet == "Fiber optic"], [25, 55], 0)
               + 5 * sum((a == "Yes").astype(int) for a in [security, backup, protection, tech])
               + 8 * ((tv == "Yes").astype(int) + (movies == "Yes").astype(int))
               + rng.normal(0, 3, n))
    monthly = np.round(np.clip(monthly, 18, None), 2)
    total = np.round(monthly * tenure * rng.uniform(0.97, 1.03, n), 2)

    logit = (-1.0 + 1.2 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
             + 0.8 * (internet == "Fiber optic") - 0.035 * tenure + 0.012 * (monthly - 65)
             + 0.45 * (payment == "Electronic check") + 0.35 * senior
             - 0.5 * (tech == "Yes") - 0.4 * (security == "Yes") + 0.2 * (paperless == "Yes"))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ids = [f"{i + 1000:04d}-{''.join(rng.choice(letters, 5))}" for i in range(n)]
    df = pd.DataFrame({
        "customerID": ids, "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        # Like the real dataset: TotalCharges is text and blank for tenure == 0
        "TotalCharges": np.where(tenure == 0, " ", total.astype(str)),
        "Churn": churn,
    })
    return pd.concat([df, df.iloc[:5]], ignore_index=True)  # 5 duplicate rows


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    make_sample().to_csv("data/sample_telco_churn.csv", index=False)
    print("Saved data/sample_telco_churn.csv")
