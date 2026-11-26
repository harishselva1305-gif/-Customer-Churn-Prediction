"""Customer Churn Prediction & Customer Retention Analytics - Streamlit app.

Run with:  streamlit run app.py
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.tree import plot_tree

import ml_utils as mu
from generate_sample_data import make_sample

st.set_page_config(page_title="Churn Prediction & Retention Analytics", page_icon="📉", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")

PALETTE = {"No": "#2E86AB", "Yes": "#E4572E"}
RISK_COLORS = {"Low": "#2E9E5B", "Medium": "#F2A900", "High": "#D64545"}
MODEL_PATH = "models/churn_models.joblib"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"

st.markdown("""
<style>
.kpi{background:#fff;border:1px solid #e6e9ef;border-left:6px solid var(--c,#2E86AB);border-radius:10px;
     padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:10px}
.kpi-label{color:#6b7280;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.kpi-value{color:#111827;font-size:1.7rem;font-weight:700;line-height:1.2}
.kpi-sub{color:#6b7280;font-size:.75rem}
.risk-box{border-radius:10px;padding:14px 18px;color:#fff;font-size:1.1rem;font-weight:600}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def kpi(col, label, value, sub="", color="#2E86AB"):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def init_state():
    defaults = {"raw_df": None, "raw_name": None, "upload_sig": None, "clean_df": None, "clean_log": None,
                "clean_before": None, "clean_after": None, "clean_info": None, "validation": None,
                "bundle": None, "chosen_model": None, "remove_dups": True}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def run_cleaning():
    df = st.session_state.raw_df
    st.session_state.validation = mu.validate_dataset(df)
    st.session_state.bundle = None  # models are stale once the data changes
    if not st.session_state.validation["ok"]:
        st.session_state.clean_df = None
        return
    (st.session_state.clean_df, st.session_state.clean_log, st.session_state.clean_before,
     st.session_state.clean_after, st.session_state.clean_info) = mu.clean_data(df, st.session_state.remove_dups)


def set_raw(df, name):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    st.session_state.raw_df, st.session_state.raw_name = df, name
    run_cleaning()


def toggle_dups():
    st.session_state.remove_dups = st.session_state.dups_widget
    run_cleaning()


def need_data():
    if st.session_state.clean_df is None:
        st.warning("No valid dataset loaded yet. Go to **📂 Data Upload** first.")
        st.stop()
    return st.session_state.clean_df


def need_models():
    if st.session_state.bundle is None:
        st.warning("No trained models yet. Go to **⚙️ Feature Engineering & Training** and train the models "
                   "(or load saved models from the sidebar).")
        st.stop()
    return st.session_state.bundle


def active_pipeline():
    b = st.session_state.bundle
    name = st.session_state.chosen_model or b["recommended"]
    return name, b["results"][name]["pipeline"]


def thresholds():
    return st.session_state.low_pct / 100, st.session_state.high_pct / 100


def churn_rate_by(df, col):
    return df.groupby(col, observed=True)["Churn"].apply(lambda s: (s == "Yes").mean() * 100)


def eda_frame(df):
    d = df.copy()
    if "SeniorCitizen" in d.columns:
        d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(d["SeniorCitizen"])
    return d


def cat_vs_churn(df, col, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, hue="Churn", order=order, hue_order=["No", "Yes"], palette=PALETTE, ax=axes[0])
    for cont in axes[0].containers:
        axes[0].bar_label(cont, fontsize=8)
    axes[0].set(title=f"Customers by {title}", xlabel=title, ylabel="Customers")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="Churn")
    rate = churn_rate_by(df, col).reindex(order)
    sns.barplot(x=rate.index, y=rate.values, color="#E4572E", ax=axes[1])
    axes[1].bar_label(axes[1].containers[0], fmt="%.1f%%", fontsize=8)
    overall = (df["Churn"] == "Yes").mean() * 100
    axes[1].axhline(overall, ls="--", color="gray", label=f"Overall churn {overall:.1f}%")
    axes[1].set(title=f"Churn rate by {title}", xlabel=title, ylabel="Churn rate (%)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend()
    show(fig)


def num_vs_churn(df, col, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    sns.histplot(data=df, x=col, hue="Churn", hue_order=["No", "Yes"], palette=PALETTE, kde=True,
                 stat="density", common_norm=False, ax=axes[0])
    axes[0].set(title=f"{title} distribution by churn", xlabel=title, ylabel="Density")
    sns.boxplot(data=df, x="Churn", y=col, hue="Churn", order=["No", "Yes"], hue_order=["No", "Yes"],
                palette=PALETTE, legend=False, ax=axes[1])
    axes[1].set(title=f"{title} vs churn", xlabel="Churn (No = stayed, Yes = left)", ylabel=title)
    show(fig)


def insights(df):
    lines = []
    overall = (df["Churn"] == "Yes").mean() * 100
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        if col in df.columns:
            g = df.groupby(col)["Churn"].agg(n="size", rate=lambda s: (s == "Yes").mean() * 100)
            g = g[g["n"] >= 30]
            if len(g):
                top = g["rate"].idxmax()
                lines.append(f"Highest churn by **{col}**: *{top}* at **{g.loc[top, 'rate']:.1f}%** "
                             f"(overall {overall:.1f}%).")
    return lines


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard():
    st.title("📉 Customer Churn Prediction & Retention Analytics")
    df = eda_frame(need_data())
    with st.expander("🔎 Filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        mask = pd.Series(True, index=df.index)
        for col, label, box in [("Contract", "Contract", c1), ("InternetService", "Internet service", c2),
                                ("gender", "Gender", c3)]:
            if col in df.columns:
                sel = box.multiselect(label, sorted(df[col].dropna().unique()))
                if sel:
                    mask &= df[col].isin(sel)
        lo, hi = int(df["tenure"].min()), int(df["tenure"].max())
        if lo < hi:
            rng = c4.slider("Tenure (months)", lo, hi, (lo, hi))
            mask &= df["tenure"].between(*rng)
    f = df[mask]
    if f.empty:
        st.warning("No customers match the filters.")
        st.stop()

    total, churned = len(f), int((f["Churn"] == "Yes").sum())
    rate = churned / total * 100
    high_txt, high_sub = "-", "Train a model to enable"
    scored = None
    if st.session_state.bundle is not None:
        name, pipe = active_pipeline()
        low, high = thresholds()
        scored = mu.score_customers(st.session_state.clean_df.loc[f.index], pipe,
                                      st.session_state.bundle["plan"]["feature_cols"], low, high)
        high_txt, high_sub = f"{(scored['RiskLevel'] == 'High').sum():,}", f"{name}, probability >= {high:.0%}"

    r1 = st.columns(4)
    kpi(r1[0], "Total customers", f"{total:,}", "in current filter")
    kpi(r1[1], "Active customers", f"{total - churned:,}", "stayed", "#2E9E5B")
    kpi(r1[2], "Churned customers", f"{churned:,}", "left the company", "#D64545")
    kpi(r1[3], "Churn rate", f"{rate:.1f}%", "churned / total", "#D64545")
    r2 = st.columns(4)
    kpi(r2[0], "Retention rate", f"{100 - rate:.1f}%", "1 - churn rate", "#2E9E5B")
    kpi(r2[1], "Avg monthly charges", f"${f['MonthlyCharges'].mean():,.2f}", "", "#7B61FF")
    kpi(r2[2], "Avg tenure", f"{f['tenure'].mean():.1f} mo", "", "#7B61FF")
    kpi(r2[3], "High-risk customers", high_txt, high_sub, "#F2A900")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.pie([total - churned, churned], labels=["Stayed", "Churned"], colors=[PALETTE["No"], PALETTE["Yes"]],
               autopct="%1.1f%%", startangle=90, wedgeprops={"width": 0.45, "edgecolor": "white"})
        ax.set_title("Churn distribution")
        show(fig)
    with c2:
        if "Contract" in f.columns:
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            r = churn_rate_by(f, "Contract")
            sns.barplot(x=r.index, y=r.values, color="#E4572E", ax=ax)
            ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=8)
            ax.set(title="Churn rate by contract", xlabel="", ylabel="Churn rate (%)")
            ax.tick_params(axis="x", rotation=15)
            show(fig)
    with c3:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        r = churn_rate_by(f.assign(Bucket=mu.tenure_bucket(f["tenure"])), "Bucket")
        sns.barplot(x=r.index.astype(str), y=r.values, color="#2E86AB", ax=ax)
        ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=8)
        ax.set(title="Churn rate by tenure", xlabel="", ylabel="Churn rate (%)")
        show(fig)

    c4, c5 = st.columns(2)
    with c4:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=f, x="Churn", y="MonthlyCharges", hue="Churn", order=["No", "Yes"],
                    hue_order=["No", "Yes"], palette=PALETTE, legend=False, ax=ax)
        ax.set(title="Monthly charges vs churn", xlabel="Churn", ylabel="Monthly charges ($)")
        show(fig)
    with c5:
        if scored is not None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            cnt = scored["RiskLevel"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            sns.barplot(x=cnt.index, y=cnt.values, hue=cnt.index, palette=RISK_COLORS, legend=False, ax=ax)
            ax.bar_label(ax.containers[0], fontsize=9)
            ax.set(title="Customers by predicted risk level", xlabel="Risk level", ylabel="Customers")
            show(fig)
            st.caption("Scores are computed on all customers in the file, including those the model trained on, "
                       "so they are slightly optimistic compared with brand-new customers.")
        elif "PaymentMethod" in f.columns:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            r = churn_rate_by(f, "PaymentMethod").sort_values()
            sns.barplot(x=r.values, y=r.index, color="#E4572E", ax=ax)
            ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=8)
            ax.set(title="Churn rate by payment method", xlabel="Churn rate (%)", ylabel="")
            show(fig)

    st.subheader("💡 Data-driven insights")
    for line in insights(f):
        st.markdown(f"- {line}")


def page_upload():
    st.header("📂 Data Upload & Preview")
    up = st.file_uploader("Upload a customer CSV file", type=["csv"])
    if up is not None:
        sig = f"{up.name}-{up.size}"
        if st.session_state.upload_sig != sig:
            try:
                set_raw(pd.read_csv(up), up.name)
                st.session_state.upload_sig = sig
            except Exception as e:
                st.error(f"Could not read this file as CSV: {e}")
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Load synthetic demo data"):
        set_raw(make_sample(), "synthetic_demo_data.csv")
        st.session_state.upload_sig = None
    if os.path.exists(DATA_PATH) and b2.button("Load Telco file from data/ folder"):
        set_raw(pd.read_csv(DATA_PATH), os.path.basename(DATA_PATH))
        st.session_state.upload_sig = None

    df = st.session_state.raw_df
    if df is None:
        st.info(f"""**No dataset loaded.** To use the real data:
1. Download the *Telco Customer Churn* dataset from Kaggle: {KAGGLE_URL}
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` above (or place it in the `data/` folder).

Want to try the app first? Click **Load synthetic demo data** - it mimics the Telco columns but is *not* real data.""")
        return
    if st.session_state.raw_name == "synthetic_demo_data.csv":
        st.warning("You are using SYNTHETIC demo data. Results are for testing the app only.")

    st.success(f"Loaded **{st.session_state.raw_name}**")
    q = mu.quality_snapshot(df)
    c = st.columns(4)
    c[0].metric("Rows", f"{q['Rows']:,}")
    c[1].metric("Columns", q["Columns"])
    c[2].metric("Duplicate rows", q["Duplicate rows"])
    c[3].metric("Missing cells", q["Missing cells"])

    v = st.session_state.validation
    for e in v["errors"]:
        st.error(e)
    for w in v["warnings"]:
        st.warning(w)
    if v["ok"]:
        st.success("Dataset validation passed.")

    t1, t2, t3 = st.tabs(["Preview", "Columns & data types", "Missing values"])
    with t1:
        n = st.slider("Rows to preview", 5, 100, 10)
        st.dataframe(df.head(n), hide_index=True)
    with t2:
        info = pd.DataFrame({"Column": df.columns, "Data type": df.dtypes.astype(str).values,
                             "Non-null": df.notna().sum().values, "Unique values": df.nunique().values})
        st.dataframe(info, hide_index=True)
    with t3:
        miss = df.isna().sum()
        blanks = pd.Series({c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns})
        tbl = pd.DataFrame({"Column": df.columns, "Missing (NaN)": miss.values, "Blank text": blanks.values})
        tbl["Missing %"] = (tbl["Missing (NaN)"] / len(df) * 100).round(2)
        st.dataframe(tbl[(tbl["Missing (NaN)"] > 0) | (tbl["Blank text"] > 0)], hide_index=True)
        st.caption("Blank text (e.g. a space in TotalCharges) is treated as missing during cleaning.")
    st.download_button("⬇️ Download uploaded dataset", csv_bytes(df), file_name=st.session_state.raw_name)


def page_cleaning():
    st.header("🧹 Data Cleaning")
    if st.session_state.raw_df is None:
        st.warning("Upload a dataset first.")
        st.stop()
    df = need_data()
    st.checkbox("Remove exact duplicate rows", value=st.session_state.remove_dups,
                key="dups_widget", on_change=toggle_dups)
    st.caption("Cleaning re-runs automatically. If you change it after training, retrain the models.")

    st.subheader("Before vs after")
    b, a = st.session_state.clean_before, st.session_state.clean_after
    st.dataframe(pd.DataFrame({"Metric": list(b), "Before": list(b.values()), "After": list(a.values())}),
                 hide_index=True)

    st.subheader("Cleaning operations performed")
    st.dataframe(st.session_state.clean_log, hide_index=True)

    info = st.session_state.clean_info
    if len(info["removed_rows"]):
        with st.expander(f"Rows removed ({len(info['removed_rows'])}) - nothing is discarded silently"):
            st.dataframe(info["removed_rows"], hide_index=True)
            st.download_button("⬇️ Download removed rows", csv_bytes(info["removed_rows"]),
                               file_name="removed_rows.csv")
    with st.expander("Columns not used for modelling"):
        for col, why in info["unnecessary"]:
            st.markdown(f"- **{col}** - {why}")
    st.subheader("Cleaned data preview")
    st.dataframe(df.head(20), hide_index=True)
    st.download_button("⬇️ Download cleaned dataset", csv_bytes(df), file_name="cleaned_churn_data.csv")


def page_eda():
    st.header("🔍 Exploratory Data Analysis")
    df = eda_frame(need_data())
    tabs = st.tabs(["Churn overview", "Categorical vs churn", "Numeric vs churn", "Retention", "Correlation"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            sns.countplot(data=df, x="Churn", hue="Churn", order=["No", "Yes"], palette=PALETTE, legend=False, ax=ax)
            ax.bar_label(ax.containers[0])
            ax.bar_label(ax.containers[1])
            ax.set(title="Churn distribution", xlabel="Churn (Yes = left)", ylabel="Customers")
            show(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            sns.histplot(df["tenure"], bins=30, kde=True, color="#2E86AB", ax=ax)
            ax.set(title="Customer tenure distribution", xlabel="Tenure (months)", ylabel="Customers")
            show(fig)
        fig, ax = plt.subplots(figsize=(10, 3.4))
        sns.histplot(df["MonthlyCharges"], bins=30, kde=True, color="#7B61FF", ax=ax)
        ax.set(title="Monthly charges distribution", xlabel="Monthly charges ($)", ylabel="Customers")
        show(fig)
    with tabs[1]:
        for col, title in [("gender", "Gender"), ("SeniorCitizen", "Senior citizen"), ("Contract", "Contract type"),
                           ("PaymentMethod", "Payment method"), ("InternetService", "Internet service")]:
            if col in df.columns:
                cat_vs_churn(df, col, title)
        other = [c for c in df.columns if df[c].dtype == object and c not in ("customerID", "Churn", "gender",
                 "SeniorCitizen", "Contract", "PaymentMethod", "InternetService") and df[c].nunique() <= 8]
        if other:
            extra = st.selectbox("Explore another feature", other)
            cat_vs_churn(df, extra, extra)
    with tabs[2]:
        for col, title in [("tenure", "Tenure (months)"), ("MonthlyCharges", "Monthly charges ($)"),
                           ("TotalCharges", "Total charges ($)")]:
            if col in df.columns:
                num_vs_churn(df, col, title)
    with tabs[3]:
        d = df.assign(Bucket=mu.tenure_bucket(df["tenure"]))
        c1, c2 = st.columns(2)
        with c1:
            r = 100 - churn_rate_by(d, "Bucket")
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            sns.barplot(x=r.index.astype(str), y=r.values, color="#2E9E5B", ax=ax)
            ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=9)
            ax.set(title="Retention rate by tenure group", xlabel="Tenure group", ylabel="Retention rate (%)",
                   ylim=(0, 105))
            show(fig)
        with c2:
            if "Contract" in df.columns:
                r = 100 - churn_rate_by(df, "Contract")
                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                sns.barplot(x=r.index, y=r.values, color="#2E86AB", ax=ax)
                ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=9)
                ax.set(title="Retention rate by contract", xlabel="Contract", ylabel="Retention rate (%)",
                       ylim=(0, 105))
                show(fig)
        st.caption("Retention rate = share of customers who did NOT churn.")
    with tabs[4]:
        base = mu.plan_features(need_data())
        X = need_data()[base["feature_cols"]]
        Xm = mu.BinaryYesNoEncoder().fit(X).transform(X)
        num = Xm[[c for c in Xm.columns if pd.api.types.is_numeric_dtype(Xm[c])]].copy()
        num["Churn (1=Yes)"] = (need_data()["Churn"] == "Yes").astype(int)
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(9, 6.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=.5, ax=ax)
        ax.set_title("Correlation heatmap (numeric and Yes/No features)")
        show(fig)
        st.markdown("**Correlation with churn**")
        st.dataframe(corr["Churn (1=Yes)"].drop("Churn (1=Yes)").sort_values(key=abs, ascending=False)
                     .round(3).rename("Correlation").reset_index().rename(columns={"index": "Feature"}),
                     hide_index=True)
        st.caption("Correlation shows association, not cause.")


def page_training():
    st.header("⚙️ Feature Engineering & Model Training")
    df = need_data()
    plan = mu.plan_features(df)
    st.subheader("1. Feature engineering plan")
    rows = ([(c, "Excluded", "Identifier / constant - not used for training") for c in plan["excluded"]
             if c != "Churn"]
            + [("Churn", "Target", "1 = churned, 0 = stayed")]
            + [(c, "Yes/No -> 1/0", "Binary encoding") for c in plan["binary_cols"]]
            + [(c, "Numeric", "Median imputation + StandardScaler") for c in plan["num_cols"]
               if c not in plan["binary_cols"]]
            + [(c, "Categorical", "Most-frequent imputation + One-Hot Encoding") for c in plan["cat_cols"]])
    st.dataframe(pd.DataFrame(rows, columns=["Column", "Treatment", "Details"]), hide_index=True)
    st.info("**No data leakage:** the data is split *first*. Imputers, scalers and encoders live inside a scikit-learn "
            "`Pipeline`/`ColumnTransformer` and are fitted on the training set only, then reused unchanged for the "
            "test set and for new predictions.")

    st.subheader("2. Train-test split & training")
    c1, c2 = st.columns(2)
    depth = c1.slider("Decision Tree max depth", 2, 15, 5)
    balance = c2.checkbox("Balance classes (class_weight='balanced')", value=True,
                          help="Gives churners more weight so the models do not ignore the minority class.")
    st.caption("Split: 80% train / 20% test, random_state = 42, stratified by Churn.")
    if st.button("🚀 Train models", type="primary"):
        with st.spinner("Training Logistic Regression and Decision Tree..."):
            try:
                bundle = mu.train_models(df, max_depth=depth, balance=balance)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.session_state.bundle = bundle
        st.session_state.chosen_model = bundle["recommended"]
        os.makedirs("models", exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        st.success(f"Models trained and saved to `{MODEL_PATH}`.")

    b = st.session_state.bundle
    if b:
        s = b["split"]
        c = st.columns(4)
        c[0].metric("Training rows", f"{s['train']:,}")
        c[1].metric("Testing rows", f"{s['test']:,}")
        c[2].metric("Churn rate (train)", f"{s['train_churn']:.1%}")
        c[3].metric("Churn rate (test)", f"{s['test_churn']:.1%}")
        st.dataframe(pd.DataFrame({n: r["metrics"] for n, r in b["results"].items()}).T.round(3))
        st.success("Next: open **📈 Model Evaluation** to compare models in detail and choose one for predictions.")


def page_evaluation():
    st.header("📈 Model Evaluation")
    b = need_models()
    res = b["results"]
    names = list(res)
    tabs = st.tabs(["Comparison", "Curves", "Logistic Regression insights", "Decision Tree insights",
                    "Business impact"])
    with tabs[0]:
        mt = pd.DataFrame({n: r["metrics"] for n, r in res.items()}).T
        st.dataframe(mt.round(3))
        fig, ax = plt.subplots(figsize=(9, 3.8))
        mt[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].T.plot(
            kind="bar", ax=ax, color=["#2E86AB", "#E4572E"])
        ax.set(title="Model comparison", ylabel="Score", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Model")
        show(fig)
        c = st.columns(2)
        for col, n in zip(c, names):
            with col:
                fig, ax = plt.subplots(figsize=(4.2, 3.6))
                sns.heatmap(res[n]["confusion"], annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"], ax=ax)
                ax.set(title=f"{n}\nConfusion matrix (test set)", xlabel="Predicted", ylabel="Actual")
                show(fig)
                with st.expander(f"Classification report - {n}"):
                    st.code(res[n]["report"])
        best, text = mu.recommend_model(res)
        st.info("🏆 **Recommendation.** " + text + " Always weigh this against your business costs (see Business impact).")
        with st.expander("📖 What do these metrics mean?"):
            st.markdown("""
- **Accuracy** - share of all predictions that were right. Misleading here: if 73% of customers stay, a model that always says "stay" scores 73%.
- **Precision** - of the customers we *flagged as churners*, how many really churned. Low precision = wasted retention offers.
- **Recall** - of the customers who *really churned*, how many we caught. Low recall = churners slip away unnoticed. Usually the most important metric for retention.
- **F1 score** - one number that balances precision and recall.
- **ROC-AUC** - how well the model ranks churners above non-churners (0.5 = guessing, 1.0 = perfect).
- **Precision-Recall curve** - shows the precision/recall trade-off at every threshold; best for imbalanced data.
- **Confusion matrix** - counts of correct and wrong predictions (bottom-left = missed churners, top-right = false alarms).
""")
        st.subheader("Choose the model used for predictions")

        def pick():
            st.session_state.chosen_model = st.session_state.model_widget
        cur = st.session_state.chosen_model if st.session_state.chosen_model in names else best
        st.selectbox("Model", names, index=names.index(cur), key="model_widget", on_change=pick)
        st.caption(f"Currently used on the Predict, Dashboard and High-Risk pages: **{st.session_state.chosen_model}**")
    with tabs[1]:
        c1, c2 = st.columns(2)
        colors = dict(zip(names, ["#2E86AB", "#E4572E"]))
        with c1:
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            for n in names:
                ax.plot(*res[n]["roc"], label=f"{n} (AUC {res[n]['metrics']['ROC-AUC']:.3f})", color=colors[n])
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
            ax.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
            ax.legend(loc="lower right")
            show(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            for n in names:
                p, r = res[n]["pr"]
                ax.plot(r, p, label=f"{n} (AP {res[n]['metrics']['PR-AUC']:.3f})", color=colors[n])
            ax.set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
            ax.legend(loc="upper right")
            show(fig)
    with tabs[2]:
        pipe = res["Logistic Regression"]["pipeline"]
        ct = mu.coefficient_table(pipe)
        top = ct.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(top["Feature"], top["Coefficient"], color=np.where(top["Coefficient"] > 0, "#E4572E", "#2E86AB"))
        ax.axvline(0, color="black", lw=.8)
        ax.set(title="Top 15 logistic regression coefficients", xlabel="Coefficient (log-odds)")
        show(fig)
        st.caption("Red = pushes toward churn, blue = pushes toward staying. Numeric features are standardised, "
                   "so coefficients are comparable. Odds ratio > 1 increases churn odds.")
        st.dataframe(ct.round(3), hide_index=True)
    with tabs[3]:
        pipe = res["Decision Tree"]["pipeline"]
        it = mu.importance_table(pipe)
        top = it.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["Feature"], top["Importance"], color="#7B61FF")
        ax.set(title="Decision tree feature importance (top 15)", xlabel="Importance")
        show(fig)
        d = st.slider("Levels of the tree to draw", 1, 4, 3)
        fig, ax = plt.subplots(figsize=(22, 9))
        plot_tree(pipe.named_steps["model"], feature_names=mu.feature_names(pipe), class_names=["Stay", "Churn"],
                  filled=True, rounded=True, max_depth=d, fontsize=9, ax=ax)
        ax.set_title("Decision tree (top levels)")
        show(fig)
        st.caption("Numeric thresholds in the tree are in *standardised* units (0 = average customer).")
    with tabs[4]:
        st.markdown("Illustrative calculation on the **test set**. Replace the assumptions with your own numbers.")
        c = st.columns(3)
        value = c[0].number_input("Revenue lost per churned customer ($)", 0, 100000, 500)
        cost = c[1].number_input("Cost of one retention offer ($)", 0, 100000, 50)
        succ = c[2].slider("Offer success rate (%)", 0, 100, 30) / 100
        rows = []
        for n in names:
            tn, fp, fn, tp = res[n]["confusion"].ravel()
            rows.append({"Model": n, "Customers contacted": int(tp + fp), "Churners reached": int(tp),
                         "Churners missed": int(fn),
                         "Estimated net benefit ($)": round(tp * (succ * value - cost) - fp * cost, 0)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Net benefit = churners reached x (success rate x value - offer cost) - wasted offers x offer cost.")


def page_predict():
    st.header("🎯 Predict Customer Churn")
    b = need_models()
    name, pipe = active_pipeline()
    low, high = thresholds()
    st.caption(f"Using model: **{name}** (change it on the Model Evaluation page). "
               f"Risk thresholds: Low < {low:.0%}, Medium < {high:.0%}, High >= {high:.0%}.")
    spec = b["input_spec"]
    order = [c for c in mu.EXPECTED_COLUMNS if c in spec] + [c for c in spec if c not in mu.EXPECTED_COLUMNS]
    vals = {}
    with st.form("predict_form"):
        cols = st.columns(3)
        for i, c in enumerate(order):
            s, box = spec[c], cols[i % 3]
            if s["kind"] == "choice":
                vals[c] = box.selectbox(c, s["options"], index=s["options"].index(s["default"]))
            elif s["kind"] == "flag":
                vals[c] = box.selectbox(c, [0, 1], index=s["options"].index(s["default"]),
                                        format_func=lambda v: "Yes" if v == 1 else "No")
            elif s["is_int"]:
                vals[c] = box.number_input(c, min_value=0, value=int(round(s["median"])), step=1)
            else:
                vals[c] = box.number_input(c, min_value=0.0, value=round(s["median"], 2), step=1.0)
        auto = st.checkbox("Calculate TotalCharges automatically (tenure x MonthlyCharges)", value=True)
        go = st.form_submit_button("🔮 Predict Churn", type="primary")
    if not go:
        return
    if auto and {"tenure", "MonthlyCharges", "TotalCharges"} <= set(vals):
        vals["TotalCharges"] = round(vals["tenure"] * vals["MonthlyCharges"], 2)
    row = pd.DataFrame([vals])[b["plan"]["feature_cols"]]
    p = float(pipe.predict_proba(row)[0, 1])
    level = mu.risk_level(p, low, high)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn prediction", "Yes - likely to leave" if p >= mu.PRED_THRESHOLD else "No - likely to stay")
    c2.metric("Churn probability / risk score", f"{p:.1%}")
    c3.markdown(f'<div class="risk-box" style="background:{RISK_COLORS[level]}">Risk level: {level}</div>',
                unsafe_allow_html=True)
    st.progress(min(max(p, 0.0), 1.0))

    base, extra = mu.retention_actions(level, vals)
    st.subheader("Retention recommendation")
    for a in base:
        st.markdown(f"- {a}")
    if extra:
        st.markdown("**Suggestions based on this customer's profile** (rule-based):")
        for a in extra:
            st.markdown(f"- {a}")

    st.subheader("Important factors")
    if name == "Logistic Regression":
        ex = mu.explain_logistic(pipe, row).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh(ex["Feature"], ex["Contribution"], color=np.where(ex["Contribution"] > 0, "#E4572E", "#2E86AB"))
        ax.axvline(0, color="black", lw=.8)
        ax.set(title="What moved this customer's score (log-odds)", xlabel="<- toward staying | toward churn ->")
        show(fig)
        st.caption("Red bars raise this customer's churn risk; blue bars lower it.")
    else:
        it = mu.importance_table(pipe).head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh(it["Feature"], it["Importance"], color="#7B61FF")
        ax.set(title="Most important features for the Decision Tree (global)", xlabel="Importance")
        show(fig)
        st.caption("Decision Tree importance is global (all customers). Switch to Logistic Regression for "
                   "per-customer explanations.")
    with st.expander("Compare both models for this customer"):
        st.dataframe(pd.DataFrame({"Model": list(b["results"]), "Churn probability": [
            f"{r['pipeline'].predict_proba(row)[0, 1]:.1%}" for r in b["results"].values()]}), hide_index=True)


def page_segmentation():
    st.header("👥 Customer Segmentation (K-Means)")
    df = need_data()
    feats = mu.segmentation_features(df)
    c1, c2 = st.columns([1, 2])
    k = c1.slider("Number of clusters (k)", 2, 8, 4)
    c2.markdown(f"**Features used (standardised):** {', '.join(feats.columns)}")
    km = mu.run_kmeans(feats, k)
    labels = km["labels"]
    churn = (df["Churn"] == "Yes").astype(int).values
    prof = mu.cluster_profile(feats, labels, churn)

    m = st.columns(3)
    m[0].metric("Clusters", k)
    m[1].metric("Silhouette score", f"{km['silhouette']:.3f}", help="Closer to 1 = better separated clusters")
    m[2].metric("Customers", f"{len(df):,}")

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        cnt = pd.Series(labels).value_counts().sort_index()
        sns.barplot(x=cnt.index.astype(str), y=cnt.values, hue=cnt.index.astype(str), legend=False,
                    palette="Set2", ax=ax)
        ax.bar_label(ax.containers[0], fontsize=9)
        ax.set(title="Customers per cluster", xlabel="Cluster ID", ylabel="Customers")
        show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        sns.scatterplot(x=km["coords"][:, 0], y=km["coords"][:, 1], hue=pd.Series(labels).astype(str),
                        palette="Set2", s=14, alpha=.7, ax=ax)
        ax.set(title="Clusters projected to 2D (PCA)", xlabel="Component 1", ylabel="Component 2")
        ax.legend(title="Cluster", markerscale=2)
        show(fig)
    st.subheader("Cluster profiles")
    st.dataframe(prof.reset_index().rename(columns={"index": "Cluster ID", "Cluster": "Cluster ID"}), hide_index=True)
    st.subheader("Interpretation (generated from the statistics above)")
    for cid, text in mu.describe_clusters(feats, labels, churn).items():
        st.markdown(f"- **Cluster {cid}:** {text}")
    if st.checkbox("Show elbow chart (helps choose k)"):
        e = mu.elbow(feats)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(e["k"], e["Inertia"], marker="o", color="#2E86AB")
        ax.set(title="Elbow method", xlabel="Number of clusters (k)", ylabel="Inertia")
        show(fig)
    st.download_button("⬇️ Download customers with cluster IDs",
                       csv_bytes(df.assign(Cluster=labels)), file_name="customers_with_clusters.csv")


def page_high_risk():
    st.header("🚨 High-Risk Customers")
    df = need_data()
    b = need_models()
    name, pipe = active_pipeline()
    low, high = thresholds()
    sc = mu.score_customers(df, pipe, b["plan"]["feature_cols"], low, high)
    c = st.columns(3)
    for col, lvl in zip(c, ["High", "Medium", "Low"]):
        kpi(col, f"{lvl}-risk customers", f"{(sc['RiskLevel'] == lvl).sum():,}", f"model: {name}", RISK_COLORS[lvl])

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    levels = f1.multiselect("Risk levels to show", ["High", "Medium", "Low"], default=["High"])
    search = f2.text_input("Search customer ID")
    table = pd.DataFrame({
        "Customer ID": sc[mu.ID_COL], "Tenure (months)": sc["tenure"],
        "Contract": sc["Contract"] if "Contract" in sc else "n/a",
        "Monthly Charges": sc["MonthlyCharges"], "Total Charges": sc["TotalCharges"],
        "Churn Probability (%)": (sc["ChurnProbability"] * 100).round(1), "Risk Level": sc["RiskLevel"]})
    sort_by = f3.selectbox("Sort by", list(table.columns), index=5)
    asc = f4.checkbox("Ascending", value=False)
    t = table[table["Risk Level"].isin(levels)]
    if search:
        t = t[t["Customer ID"].astype(str).str.contains(search, case=False)]
    t = t.sort_values(sort_by, ascending=asc)
    st.caption(f"{len(t):,} customers shown. You can also click any column header to sort.")
    st.dataframe(t, hide_index=True, column_config={
        "Churn Probability (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        "Monthly Charges": st.column_config.NumberColumn(format="$%.2f"),
        "Total Charges": st.column_config.NumberColumn(format="$%.2f")})
    st.download_button("⬇️ Download this table", csv_bytes(t), file_name="high_risk_customers.csv")
    st.caption("Note: scores include customers the model trained on, so they are slightly optimistic. "
               "For real campaigns, score current customers who were not in the training file.")


def page_regression():
    st.header("📐 Linear Regression Demo")
    st.info("This section predicts a **number** (a continuous value such as MonthlyCharges). "
            "It is separate from churn prediction, which predicts a **category** (Churn / Stay).")
    df = need_data()
    target = st.selectbox("Value to predict", [c for c in ["MonthlyCharges", "TotalCharges"] if c in df.columns])
    options = [c for c in df.columns if c not in (mu.ID_COL, "Churn", target)]
    default = [c for c in (["tenure", "InternetService", "PhoneService", "MultipleLines", "OnlineSecurity",
                            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
                           if target == "MonthlyCharges" else ["tenure", "MonthlyCharges", "Contract"])
               if c in options]
    feats = st.multiselect("Input features", options, default=default)
    if not feats:
        st.warning("Select at least one feature.")
        return
    r = mu.train_linear_regression(df, target, feats)
    c = st.columns(4)
    c[0].metric("R² (test)", f"{r['R2']:.3f}", help="1.0 = perfect, 0 = no better than predicting the average")
    c[1].metric("MAE", f"${r['MAE']:,.2f}", help="Average absolute error")
    c[2].metric("RMSE", f"${r['RMSE']:,.2f}", help="Error that penalises big misses")
    c[3].metric("Train / test rows", f"{r['train']:,} / {r['test']:,}")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5.5, 4.3))
        ax.scatter(r["y_test"], r["pred"], s=10, alpha=.5, color="#2E86AB", label="Customers")
        lim = [min(r["y_test"].min(), r["pred"].min()), max(r["y_test"].max(), r["pred"].max())]
        ax.plot(lim, lim, "--", color="#E4572E", label="Perfect prediction")
        ax.set(title=f"Actual vs predicted {target}", xlabel="Actual", ylabel="Predicted")
        ax.legend()
        show(fig)
    with b:
        top = r["coefs"].head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5.5, 4.3))
        ax.barh(top["Feature"], top["Coefficient"], color="#7B61FF")
        ax.set(title="Largest coefficients", xlabel="Effect on prediction (per std. dev. / category)")
        show(fig)
    st.caption("Regression predicts a continuous amount; it does not predict whether a customer will churn.")


# --------------------------------------------------------------------------- #
# Sidebar & routing
# --------------------------------------------------------------------------- #
init_state()
PAGES = {"📊 Dashboard": page_dashboard, "📂 Data Upload": page_upload, "🧹 Data Cleaning": page_cleaning,
         "🔍 EDA": page_eda, "⚙️ Feature Engineering & Training": page_training,
         "📈 Model Evaluation": page_evaluation, "🎯 Predict Churn": page_predict,
         "👥 Customer Segmentation": page_segmentation, "🚨 High-Risk Customers": page_high_risk,
         "📐 Linear Regression Demo": page_regression}

with st.sidebar:
    st.title("📉 Churn Analytics")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown("**Risk thresholds**")
    st.slider("Medium risk starts at (%)", 5, 90, 30, key="low_pct")
    st.slider("High risk starts at (%)", 10, 95, 60, key="high_pct")
    if st.session_state.low_pct >= st.session_state.high_pct:
        st.error("Medium threshold must be below the High threshold.")
        st.stop()
    st.divider()
    st.caption(f"Data: {st.session_state.raw_name or 'none'}")
    st.caption(f"Models: {'trained' if st.session_state.bundle else 'not trained'}")
    if os.path.exists(MODEL_PATH) and st.button("Load saved models"):
        try:
            st.session_state.bundle = joblib.load(MODEL_PATH)
            st.session_state.chosen_model = st.session_state.bundle["recommended"]
            st.rerun()
        except Exception as e:
            st.error(f"Could not load saved models: {e}")

PAGES[page]()
