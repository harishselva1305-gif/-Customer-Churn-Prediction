"""Data cleaning, feature engineering, modelling and business-rule helpers.

Kept in its own module (not inside app.py) so that joblib can reliably
save / load the trained pipelines, which contain the custom transformer below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
PRED_THRESHOLD = 0.5
REQUIRED_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
EXPECTED_COLUMNS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                    "MonthlyCharges", "TotalCharges", "Churn"]
SERVICE_COLUMNS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CHURN_MAP = {"yes": "Yes", "no": "No", "1": "Yes", "0": "No", "1.0": "Yes", "0.0": "No",
             "true": "Yes", "false": "No"}


# --------------------------------------------------------------------------- #
# Validation & cleaning
# --------------------------------------------------------------------------- #
def validate_dataset(df: pd.DataFrame) -> dict:
    errors, warnings = [], []
    if df is None or df.empty:
        return {"errors": ["The file is empty."], "warnings": [], "ok": False}
    cols = set(df.columns.astype(str).str.strip())
    missing_req = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing_req:
        errors.append(f"Missing required column(s): {', '.join(missing_req)}.")
    missing_opt = [c for c in EXPECTED_COLUMNS if c not in cols and c not in REQUIRED_COLUMNS]
    if missing_opt:
        warnings.append("Optional Telco columns not found (the app adapts, but some charts/features "
                        "will be unavailable): " + ", ".join(missing_opt))
    if TARGET in cols:
        vals = df[TARGET].dropna().astype(str).str.strip().str.lower().unique()
        bad = sorted(set(vals) - set(CHURN_MAP))
        if bad:
            warnings.append(f"Unrecognised Churn values {bad[:5]} - those rows will be listed and "
                            "removed during cleaning.")
        if len({CHURN_MAP[v] for v in vals if v in CHURN_MAP}) < 2:
            errors.append("The Churn column must contain both Yes and No customers.")
    if len(df) < 100:
        warnings.append("Fewer than 100 rows - model results will be unreliable.")
    return {"errors": errors, "warnings": warnings, "ok": not errors}


def quality_snapshot(df: pd.DataFrame) -> dict:
    return {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Missing cells": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Numeric columns": int(df.select_dtypes(include="number").shape[1]),
        "Text columns": int(df.shape[1] - df.select_dtypes(include="number").shape[1]),
    }


def clean_data(df: pd.DataFrame, remove_duplicates: bool = True):
    """Clean the raw data. Returns (clean_df, log_df, before, after, info).

    Nothing is dropped silently: every removed row is kept in info['removed_rows']
    with a reason. Statistical imputation (median / most frequent) is deliberately
    NOT done here - it happens inside the model pipeline so it is fitted on the
    training split only (no data leakage).
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    before = quality_snapshot(d)
    log, removed = [], []

    def add(step, action, n, details=""):
        log.append({"Step": step, "Action": action, "Affected": int(n), "Details": details})

    # 1. whitespace / blank strings -> missing
    for c in d.columns:
        if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
            na = d[c].isna()
            s = d[c].astype(str).str.strip()
            blank = (~na) & (s == "")
            d[c] = s.where(~(na | blank), np.nan).astype(object)
            if blank.sum():
                add("Blank values", f"'{c}': blank text converted to missing", blank.sum(),
                    "Blank strings are real missing values, not valid data.")

    # 2. data types
    if "SeniorCitizen" in d.columns and not pd.api.types.is_numeric_dtype(d["SeniorCitizen"]):
        d["SeniorCitizen"] = d["SeniorCitizen"].replace({"Yes": 1, "No": 0})
    for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if c in d.columns and not pd.api.types.is_numeric_dtype(d[c]):
            orig_na = d[c].isna()
            conv = pd.to_numeric(d[c], errors="coerce")
            bad = int((conv.isna() & ~orig_na).sum())
            d[c] = conv
            add("Data types", f"'{c}' converted to numeric", len(d),
                f"{bad} value(s) were not valid numbers and became missing." if bad else "All values parsed.")

    # 3. logical fix: brand-new customers have no charges yet
    if {"tenure", "TotalCharges"} <= set(d.columns):
        m = (d["tenure"] == 0) & d["TotalCharges"].isna()
        if m.any():
            d.loc[m, "TotalCharges"] = 0.0
            add("Missing values", "TotalCharges set to 0 where tenure = 0", m.sum(),
                "A customer with 0 months of tenure has not been billed yet.")

    # 4. invalid values (negative tenure / charges)
    inv = pd.Series(False, index=d.index)
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if c in d.columns:
            inv |= d[c] < 0
    if inv.any():
        removed.append(d[inv].assign(RemovedReason="Negative tenure/charges"))
        d = d[~inv]
        add("Invalid data", "Rows with negative tenure/charges removed", inv.sum(), "Listed in the removed-rows table.")

    # 5. target column
    mapped = d[TARGET].astype(str).str.strip().str.lower().map(CHURN_MAP)
    nt = mapped.isna()
    if nt.any():
        removed.append(d[nt].assign(RemovedReason="Missing/invalid Churn label"))
        add("Target", "Rows without a valid Churn label removed", nt.sum(),
            "They cannot be used to train or evaluate a model.")
    d[TARGET] = mapped
    d = d[~nt]

    # 6. duplicates
    dup = d.duplicated()
    if remove_duplicates and dup.any():
        removed.append(d[dup].assign(RemovedReason="Exact duplicate row"))
        d = d[~dup]
        add("Duplicates", "Exact duplicate rows removed", dup.sum(), "First occurrence kept.")
    elif dup.any():
        add("Duplicates", "Duplicate rows kept (removal switched off)", dup.sum(), "")
    if ID_COL not in d.columns:
        d.insert(0, ID_COL, [f"CUST-{i + 1:05d}" for i in range(len(d))])
        add("Identifiers", "customerID column was missing - generated", len(d), "")
    elif d[ID_COL].duplicated().any():
        add("Identifiers", "Repeated customerID values found (rows differ, so kept)",
            d[ID_COL].duplicated().sum(), "Check whether the same customer appears more than once.")

    d = d.reset_index(drop=True)

    # 7. columns that are unnecessary for modelling (reported, not deleted)
    unnecessary = []
    for c in d.columns:
        if c == TARGET:
            continue
        if c == ID_COL:
            unnecessary.append((c, "Identifier - excluded from model training (would cause overfitting)"))
        elif d[c].nunique(dropna=True) <= 1:
            unnecessary.append((c, "Constant column - carries no information"))
        elif d[c].isna().mean() > 0.5:
            unnecessary.append((c, "More than 50% missing"))

    after = quality_snapshot(d)
    remaining_missing = d.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if len(remaining_missing):
        add("Missing values", "Remaining missing values left for in-pipeline imputation",
            remaining_missing.sum(), ", ".join(f"{k}: {v}" for k, v in remaining_missing.items()))
    if not log:
        add("Summary", "No problems found", 0, "Data was already clean.")
    info = {
        "removed_rows": pd.concat(removed) if removed else pd.DataFrame(),
        "unnecessary": unnecessary,
        "remaining_missing": remaining_missing,
    }
    return d, pd.DataFrame(log), before, after, info


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class BinaryYesNoEncoder(BaseEstimator, TransformerMixin):
    """Map columns whose training values are only Yes/No to 1/0.

    The set of binary columns is learned in fit(), so prediction rows are
    transformed exactly the way the training data was.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.binary_cols_ = [
            c for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c]) and X[c].notna().any()
            and set(X[c].dropna().unique()) <= {"Yes", "No"}
        ]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.binary_cols_:
            if c in X.columns:
                X[c] = X[c].map({"Yes": 1, "No": 0}).astype(float)
        return X


def plan_features(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c not in (TARGET, ID_COL) and df[c].nunique(dropna=True) > 1]
    X = df[feature_cols]
    enc = BinaryYesNoEncoder().fit(X)
    Xm = enc.transform(X)
    num_cols = [c for c in Xm.columns if pd.api.types.is_numeric_dtype(Xm[c])]
    cat_cols = [c for c in Xm.columns if c not in num_cols]
    excluded = [c for c in df.columns if c not in feature_cols]
    return {"feature_cols": feature_cols, "excluded": excluded, "binary_cols": enc.binary_cols_,
            "num_cols": num_cols, "cat_cols": cat_cols}


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))])
    return ColumnTransformer([("num", num, list(num_cols)), ("cat", cat, list(cat_cols))])


def make_pipeline(model, num_cols, cat_cols) -> Pipeline:
    return Pipeline([("binary", BinaryYesNoEncoder()),
                     ("prep", build_preprocessor(num_cols, cat_cols)),
                     ("model", model)])


def feature_names(pipe: Pipeline) -> list[str]:
    names = pipe.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def _input_spec(X: pd.DataFrame, binary_cols) -> dict:
    spec = {}
    for c in X.columns:
        s = X[c].dropna()
        if c in binary_cols:
            spec[c] = {"kind": "choice", "options": ["No", "Yes"], "default": s.mode().iloc[0]}
        elif pd.api.types.is_numeric_dtype(X[c]):
            if set(s.unique()) <= {0, 1}:
                spec[c] = {"kind": "flag", "options": [0, 1], "default": int(s.mode().iloc[0])}
            else:
                spec[c] = {"kind": "number", "median": float(s.median()),
                           "is_int": bool((s == s.round()).all())}
        else:
            opts = sorted(s.astype(str).unique())
            spec[c] = {"kind": "choice", "options": opts, "default": s.mode().iloc[0]}
    return spec


# --------------------------------------------------------------------------- #
# Classification: training & evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= PRED_THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    return {
        "metrics": {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "PR-AUC": average_precision_score(y_test, proba),
        },
        "confusion": confusion_matrix(y_test, pred),
        "report": classification_report(y_test, pred, target_names=["Stay", "Churn"], zero_division=0),
        "roc": (fpr, tpr), "pr": (prec, rec),
    }


def train_models(df: pd.DataFrame, max_depth: int = 5, balance: bool = True) -> dict:
    plan = plan_features(df)
    X, y = df[plan["feature_cols"]], (df[TARGET] == "Yes").astype(int)
    if y.value_counts().min() < 2:
        raise ValueError("Each class needs at least 2 customers to split the data.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cw = "balanced" if balance else None
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10,
                                                class_weight=cw, random_state=RANDOM_STATE),
    }
    results = {}
    for name, model in models.items():
        pipe = make_pipeline(model, plan["num_cols"], plan["cat_cols"]).fit(X_train, y_train)
        results[name] = {"pipeline": pipe, **evaluate_model(pipe, X_test, y_test)}
    enc = BinaryYesNoEncoder().fit(X)
    return {
        "results": results, "plan": plan, "recommended": recommend_model(results)[0],
        "input_spec": _input_spec(X, enc.binary_cols_),
        "split": {"train": len(X_train), "test": len(X_test),
                  "train_churn": float(y_train.mean()), "test_churn": float(y_test.mean())},
        "settings": {"max_depth": max_depth, "balanced": balance},
    }


def recommend_model(results: dict):
    """Pick by churn F1 (tie-break: recall) - never by accuracy alone."""
    ranked = sorted(results, key=lambda n: (results[n]["metrics"]["F1 Score"],
                                            results[n]["metrics"]["Recall"]), reverse=True)
    best, other = ranked[0], ranked[1]
    b, o = results[best]["metrics"], results[other]["metrics"]
    text = (f"**{best}** gives the best balance for catching churners: F1 {b['F1 Score']:.3f} and recall "
            f"{b['Recall']:.3f}, versus F1 {o['F1 Score']:.3f} / recall {o['Recall']:.3f} for {other}. ")
    if o["Accuracy"] > b["Accuracy"]:
        text += (f"{other} has higher raw accuracy ({o['Accuracy']:.3f} vs {b['Accuracy']:.3f}), but accuracy "
                 "alone is misleading when most customers stay.")
    return best, text


# --------------------------------------------------------------------------- #
# Prediction, risk & retention logic
# --------------------------------------------------------------------------- #
def risk_level(p: float, low: float = 0.30, high: float = 0.60) -> str:
    return "Low" if p < low else ("Medium" if p < high else "High")


BASE_ACTIONS = {
    "Low": ["Maintain good service quality.", "Offer loyalty rewards when appropriate."],
    "Medium": ["Send personalised offers.", "Review customer satisfaction.", "Improve service engagement."],
    "High": ["Prioritise retention outreach.", "Provide targeted discounts or contract options.",
             "Investigate customer support issues."],
}


def retention_actions(level: str, row: dict | None = None) -> tuple[list[str], list[str]]:
    """Returns (standard actions, profile-specific suggestions)."""
    extra = []
    if row and level in ("Medium", "High"):
        if row.get("Contract") == "Month-to-month":
            extra.append("Offer a discounted 1- or 2-year contract (month-to-month customers churn most easily).")
        if "Electronic check" in str(row.get("PaymentMethod", "")):
            extra.append("Encourage automatic payment (bank transfer / card).")
        if row.get("InternetService") not in (None, "No") and row.get("TechSupport") == "No":
            extra.append("Offer a free trial of Tech Support.")
        if row.get("InternetService") not in (None, "No") and row.get("OnlineSecurity") == "No":
            extra.append("Offer a free trial of Online Security.")
        try:
            if float(row.get("tenure", 99)) < 12:
                extra.append("Run an onboarding check-in - the customer is still in their first year.")
        except (TypeError, ValueError):
            pass
    return BASE_ACTIONS[level], extra


def score_customers(df: pd.DataFrame, pipe, feature_cols, low: float, high: float) -> pd.DataFrame:
    out = df.copy()
    out["ChurnProbability"] = pipe.predict_proba(df.reindex(columns=feature_cols))[:, 1]
    out["RiskLevel"] = [risk_level(p, low, high) for p in out["ChurnProbability"]]
    return out


def explain_logistic(pipe, row: pd.DataFrame) -> pd.DataFrame:
    """Per-customer contribution (in log-odds) of each encoded feature."""
    xt = pipe.named_steps["prep"].transform(pipe.named_steps["binary"].transform(row))[0]
    contrib = xt * pipe.named_steps["model"].coef_[0]
    out = pd.DataFrame({"Feature": feature_names(pipe), "Contribution": contrib})
    return out.reindex(out["Contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def coefficient_table(pipe) -> pd.DataFrame:
    coef = pipe.named_steps["model"].coef_[0]
    t = pd.DataFrame({"Feature": feature_names(pipe), "Coefficient": coef, "Odds ratio": np.exp(coef)})
    return t.reindex(t["Coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def importance_table(pipe) -> pd.DataFrame:
    t = pd.DataFrame({"Feature": feature_names(pipe),
                      "Importance": pipe.named_steps["model"].feature_importances_})
    return t.sort_values("Importance", ascending=False).reset_index(drop=True)


def tenure_bucket(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-1, 12, 24, 48, np.inf], labels=["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"])


# --------------------------------------------------------------------------- #
# K-Means segmentation
# --------------------------------------------------------------------------- #
FRIENDLY = {"tenure": "tenure (months)", "MonthlyCharges": "monthly charges", "TotalCharges": "total charges",
            "ContractLength": "contract length (0=monthly,1=1yr,2=2yr)", "ServicesCount": "number of services"}


def segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if c in df.columns:
            f[c] = pd.to_numeric(df[c], errors="coerce")
    if "Contract" in df.columns:
        cl = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})
        if cl.notna().any():
            f["ContractLength"] = cl
    svc = [c for c in SERVICE_COLUMNS if c in df.columns]
    if svc:
        f["ServicesCount"] = sum((df[c] == "Yes").astype(int) for c in svc)
    return f.fillna(f.median())


def run_kmeans(feats: pd.DataFrame, k: int) -> dict:
    Xs = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    sil = silhouette_score(Xs, km.labels_, sample_size=min(3000, len(Xs)), random_state=RANDOM_STATE)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    return {"labels": km.labels_, "silhouette": float(sil), "coords": coords}


def elbow(feats: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    Xs = StandardScaler().fit_transform(feats)
    return pd.DataFrame({"k": list(ks), "Inertia": [
        KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit(Xs).inertia_ for k in ks]})


def cluster_profile(feats, labels, churn_flag=None) -> pd.DataFrame:
    prof = feats.assign(Cluster=labels).groupby("Cluster").mean().round(2)
    prof.insert(0, "Customers", pd.Series(labels).value_counts().sort_index())
    prof.insert(1, "Share %", (prof["Customers"] / prof["Customers"].sum() * 100).round(1))
    if churn_flag is not None:
        prof["Churn rate %"] = (pd.Series(np.asarray(churn_flag)).groupby(labels).mean() * 100).round(1)
    return prof


def describe_clusters(feats, labels, churn_flag=None) -> dict:
    """Plain-English descriptions generated ONLY from the cluster statistics."""
    mean, std = feats.mean(), feats.std().replace(0, 1)
    out = {}
    for k in sorted(set(labels)):
        m = feats[labels == k]
        z = (m.mean() - mean) / std
        parts = []
        for f in z.abs().sort_values(ascending=False).index[:3]:
            if abs(z[f]) < 0.25:
                continue
            parts.append(f"{FRIENDLY.get(f, f)} is {'higher' if z[f] > 0 else 'lower'} than average "
                         f"({m[f].mean():.1f} vs {mean[f]:.1f})")
        if not parts:
            parts = ["is close to the overall average on every segmentation feature"]
        if churn_flag is not None:
            cf = np.asarray(churn_flag)
            parts.append(f"churn rate is {cf[labels == k].mean():.1%} vs {cf.mean():.1%} overall")
        out[k] = "; ".join(parts) + "."
    return out


# --------------------------------------------------------------------------- #
# Linear regression demo
# --------------------------------------------------------------------------- #
def train_linear_regression(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    d = df.dropna(subset=[target])
    X, y = d[features], d[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    enc = BinaryYesNoEncoder().fit(X)
    Xm = enc.transform(X)
    num = [c for c in Xm.columns if pd.api.types.is_numeric_dtype(Xm[c])]
    cat = [c for c in Xm.columns if c not in num]
    pipe = make_pipeline(LinearRegression(), num, cat).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    coefs = pd.DataFrame({"Feature": feature_names(pipe), "Coefficient": pipe.named_steps["model"].coef_})
    return {"pipeline": pipe, "y_test": y_test.values, "pred": pred,
            "R2": r2_score(y_test, pred), "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "coefs": coefs.reindex(coefs["Coefficient"].abs().sort_values(ascending=False).index),
            "train": len(X_train), "test": len(X_test)}
