# Customer Churn Prediction & Customer Retention Analytics

A Streamlit app that cleans customer data, explores it, trains Logistic Regression and Decision Tree churn
models, segments customers with K-Means, and flags high-risk customers with retention recommendations.

## Quick start
From the workspace parent directory (the directory containing `churn_project`), run:

```bash
cd churn_project
pip install -r requirements.txt
streamlit run app.py
```
1. Download the **Telco Customer Churn** CSV from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Open **Data Upload** and upload `WA_Fn-UseC_-Telco-Customer-Churn.csv`
   (or put it in `data/` and click *Load Telco file from data/ folder*).
3. Work through the sidebar pages top to bottom; train models on **Feature Engineering & Training**.

No dataset yet? Click **Load synthetic demo data** (or use `data/sample_telco_churn.csv`).
It mimics the Telco columns but is **fake** - use it only to try the app.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (10 pages) |
| `ml_utils.py` | Cleaning, pipelines, models, risk rules, K-Means, linear regression |
| `generate_sample_data.py` | Creates synthetic demo data |
| `models/` | Trained models saved with Joblib (created on first training) |

## Design notes
- **No data leakage:** split happens first (80/20, `random_state=42`, stratified); imputers, scaler and one-hot
  encoder are inside a scikit-learn `Pipeline` + `ColumnTransformer` fitted on training data only.
- **Same preprocessing for prediction:** the saved pipeline includes every transformation.
- **Model choice:** recommended by churn F1 (tie-break recall), not accuracy. Class balancing is on by default.
- **Cleaning never silently drops data:** removed rows are listed and downloadable.
- **Risk levels:** Low < 30%, Medium 30-60%, High >= 60% (adjustable in the sidebar).
- Cluster descriptions are generated from the cluster statistics, not hand-written.
- Dashboard / high-risk scores include customers the model trained on, so they are slightly optimistic.
