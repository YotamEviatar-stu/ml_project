"""Experimental audit: impact of the PREPAID NONREFUNDABLE (non-affiliated) row exclusion.

Standalone sandbox script — does NOT touch Group_46_Notebook.ipynb.

Replicates the notebook's Stage 1/2 cleaning + feature engineering and the tuned
XGBoost baseline (cell 123 best params), then loops over two scenarios:

  Iteration 0 (Baseline)   : exclude the PREPAID NONREFUNDABLE + non-affiliated
                             slice (~9,695 rows), replicating notebook cell 102.
  Iteration 1 (Reinstated) : include that slice back in the modeling population.

Each iteration is evaluated with two validation strategies:
  A. Random stratified 80/20 holdout (matching notebook cell 113, random_state=42).
  B. Temporal split — train on the early period, validate on the last 3 months of
     Course_Start_Date (mimics the future test-set distribution).

Time-feature note: the raw day-index `course_start_days` is REPLACED by seasonal
components (month-of-year, day-of-week) in this audit, so the temporal split is
not corrupted by tree extrapolation on an ever-increasing day counter.

Outputs metrics to the console and writes prepaid_impact_results.md.

Run:  .venv/bin/python experimental_audit_prepaid.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "Dropped_Course"
TEMPORAL_VAL_MONTHS = 3

# Tuned params from the notebook's XGBoost RandomizedSearchCV (cell 123 output)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    colsample_bytree=0.7024273291045295,
    learning_rate=0.011474276591948267,
    max_depth=12,
    n_estimators=853,
    reg_alpha=0.05719341217670732,
    reg_lambda=0.0064102771881101635,
    subsample=0.9583054382694077,
)


# ---------------------------------------------------------------------------
# Stage 1 cleaning — identical regex/transform logic to the notebook
# ---------------------------------------------------------------------------

def clean_text_upper(series, keep_digits=True):
    pattern = r"[^A-Z0-9\s]" if keep_digits else r"[^A-Z\s]"
    raw = series.astype(str)
    cleaned = raw.str.upper().str.replace(pattern, "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned.where((cleaned != "") & series.notna(), "MISSING")


def add_clean_columns(df):
    # Origin_Country (cell 64)
    raw = df["Origin_Country"].astype(str)
    cleaned = raw.str.upper().str.replace(r"[^A-Z]", "", regex=True)
    cleaned = cleaned.replace("CN", "CHN")
    df["country_clean"] = cleaned.where(df["Origin_Country"].notna(), "MISSING")

    # Client_Category (cell 67) — note: no MISSING fallback in the notebook
    raw = df["Client_Category"].astype(str)
    cleaned = raw.str.upper().str.replace(r"[^A-Z\s]", "", regex=True)
    df["client_category_clean"] = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()

    df["payment_terms_clean"] = clean_text_upper(df["Payment_Terms"])
    df["payment_terms_bucket"] = df["payment_terms_clean"].where(
        df["payment_terms_clean"].isin(["PAY UPON START", "PREPAID NONREFUNDABLE"]), "OTHER")
    df["enrollment_type_clean"] = clean_text_upper(df["Enrollment_Type"])
    df["submission_source_clean"] = clean_text_upper(df["Submission_Source"])
    df["catering_package_clean"] = clean_text_upper(df["Catering_Package"])

    # Sentinel recodes (cells 28, 34)
    df["students_count_clean"] = df["Students_Count"].where(df["Students_Count"] != 9999)
    df["practical_hours_clean"] = df["Practical_Hours"].where(
        ~df["Practical_Hours"].isin([-5, -1, 5000, 10000]))

    df["Course_Start_Date"] = pd.to_datetime(df["Course_Start_Date"])
    return df


# ---------------------------------------------------------------------------
# Stage 2 feature engineering — replicates cells 103-106
# (category vocabularies are computed once from the FULL train file, as the
#  notebook does, so both iterations share an identical feature space)
# ---------------------------------------------------------------------------

def build_vocabularies(df):
    lab_full = df["Requested_Lab_Config"].where(df["Requested_Lab_Config"].notna(), "MISSING")
    top_lab_configs = lab_full[lab_full != "MISSING"].value_counts().head(6).index.tolist()

    country_pct = df.loc[df["country_clean"] != "MISSING", "country_clean"] \
        .value_counts(normalize=True) * 100
    top_countries = list(country_pct[country_pct >= 4.0].index)

    company_counts = df["Company_ID"].value_counts()
    big_companies = [str(int(v)) for v in company_counts[company_counts >= 30].index]
    agent_counts = df["Agent_ID"].value_counts()
    big_agents = [str(int(v)) for v in agent_counts[agent_counts >= 30].index]
    return top_lab_configs, top_countries, big_companies, big_agents


def build_features(df_model, vocabs):
    top_lab_configs, top_countries, big_companies, big_agents = vocabs

    daily_tuition_clean = df_model["Daily_Tuition_Cost"].where(df_model["Daily_Tuition_Cost"] != 5400)

    registration_days_imputed = df_model["Registration_Days_Before"].fillna(
        df_model["Registration_Days_Before"].median())
    students_count_imputed = df_model["students_count_clean"].fillna(
        df_model["students_count_clean"].median())
    practical_hours_imputed = df_model["practical_hours_clean"].fillna(
        df_model["practical_hours_clean"].median())
    daily_tuition_imputed = daily_tuition_clean.fillna(daily_tuition_clean.median())

    registration_changes_group = pd.Categorical(pd.Series(
        np.select(
            [df_model["Registration_Changes"] == 0, df_model["Registration_Changes"] == 1],
            ["0", "1"], default="2+"),
        index=df_model.index), categories=["0", "1", "2+"])

    lab_config_clean = df_model["Requested_Lab_Config"].where(
        df_model["Requested_Lab_Config"].notna(), "MISSING")
    lab_config_grouped = pd.Categorical(
        lab_config_clean.where(lab_config_clean.isin(top_lab_configs + ["MISSING"]), "OTHER"),
        categories=top_lab_configs + ["OTHER", "MISSING"])

    country_grouped = pd.Categorical(
        df_model["country_clean"].where(
            df_model["country_clean"].isin(top_countries + ["MISSING"]), "OTHER"),
        categories=top_countries + ["OTHER", "MISSING"])

    company_grouped = df_model["Company_ID"].apply(
        lambda v: "OTHER" if pd.isna(v) else str(int(v)))
    company_grouped = pd.Categorical(
        company_grouped.where(company_grouped.isin(big_companies), "OTHER"),
        categories=big_companies + ["OTHER"])

    agent_grouped = df_model["Agent_ID"].apply(lambda v: "OTHER" if pd.isna(v) else str(int(v)))
    agent_grouped = pd.Categorical(
        agent_grouped.where(agent_grouped.isin(big_agents), "OTHER"),
        categories=big_agents + ["OTHER"])

    one_hot_df = pd.get_dummies(
        pd.DataFrame({
            "registration_changes": registration_changes_group,
            "requested_lab_config": lab_config_grouped,
            "origin_country": country_grouped,
            "client_category": pd.Categorical(df_model["client_category_clean"]),
            "enrollment_type": pd.Categorical(df_model["enrollment_type_clean"]),
            "submission_source": pd.Categorical(df_model["submission_source_clean"]),
            "catering_package": pd.Categorical(df_model["catering_package_clean"]),
            "payment_terms": pd.Categorical(
                df_model["payment_terms_bucket"],
                categories=["PAY UPON START", "PREPAID NONREFUNDABLE", "OTHER"]),
            "company_id_group": company_grouped,
            "agent_id_group": agent_grouped,
        }, index=df_model.index),
        prefix_sep="__",
    ).astype(int)

    numeric_df = pd.DataFrame({
        "had_prior_dropout": (df_model["Prev_Course_Dropouts"] >= 1).astype(int),
        "attended_before": (df_model["Prev_Course_Attended"] >= 1).astype(int),
        "registration_days_before": registration_days_imputed,
        "waiting_list_days": df_model["Waiting_List_Days"],
        "has_waited": (df_model["Waiting_List_Days"] > 0).astype(int),
        "has_support_ticket": (df_model["Pre_Course_Supports_Tickets"] > 0).astype(int),
        "professionals_count": df_model["Professionals_Count"],
        "students_count": students_count_imputed,
        "has_observers": (df_model["Observers_Count"] > 0).astype(int),
        "practical_hours": practical_hours_imputed,
        "theory_hours": df_model["Theory_Hours"],
        "daily_tuition_cost": daily_tuition_imputed,
        "is_zero_cost": (daily_tuition_clean == 0).astype(int),
        "returning_client": df_model["Returning_Client"],
        "has_company_id": df_model["Company_ID"].notna().astype(int),
        "has_agent_id": df_model["Agent_ID"].notna().astype(int),
        # AUDIT CHANGE: seasonal components instead of the raw day index
        # `course_start_days`, to avoid tree extrapolation on the temporal split.
        "start_month": df_model["Course_Start_Date"].dt.month,
        "start_dayofweek": df_model["Course_Start_Date"].dt.dayofweek,
    })

    X = pd.concat([numeric_df, one_hot_df], axis=1)
    y = df_model[TARGET]
    dates = df_model["Course_Start_Date"]
    assert X.isna().sum().sum() == 0, "unexpected nulls in X"
    return X, y, dates


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(X_tr, y_tr, X_val, y_val):
    model = XGBClassifier(**XGB_PARAMS)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    tn, fp, fn, tp = confusion_matrix(y_val, (proba >= 0.5).astype(int)).ravel()
    return dict(auc=auc, fn=int(fn), fp=int(fp), tn=int(tn), tp=int(tp),
                n_train=len(X_tr), n_val=len(X_val),
                val_drop_rate=float(y_val.mean()), fit_sec=time.time() - t0)


def main():
    df = add_clean_columns(pd.read_csv("extra/Train_Data.csv"))
    test_df = add_clean_columns(pd.read_csv("to_submit/Test_Data_No_Target.csv").assign(**{TARGET: np.nan}))

    vocabs = build_vocabularies(df)

    is_prepaid = df["payment_terms_clean"] == "PREPAID NONREFUNDABLE"
    is_affiliated = df["enrollment_type_clean"] == "AFFILIATED ADMISSION"
    prepaid_slice = is_prepaid & ~is_affiliated
    exclude_unknown = df["client_category_clean"] == "UNKNOWN"

    print("=" * 78)
    print("SLICE IDENTIFICATION (train file)")
    print("=" * 78)
    print(f"rows in Train_Data.csv                       : {len(df):>7,}")
    print(f"PREPAID NONREFUNDABLE + non-affiliated slice : {int(prepaid_slice.sum()):>7,}"
          f"  (drop rate {df.loc[prepaid_slice, TARGET].mean():.4%})")
    print(f"Client_Category == UNKNOWN (always excluded) : {int(exclude_unknown.sum()):>7,}")

    # Structural cross-check against the unlabeled test file
    t_prepaid = (test_df["payment_terms_clean"] == "PREPAID NONREFUNDABLE") & \
                (test_df["enrollment_type_clean"] != "AFFILIATED ADMISSION")
    print(f"\nTest_Data_No_Target.csv rows                 : {len(test_df):>7,}")
    print(f"  in the same PREPAID+non-affiliated slice   : {int(t_prepaid.sum()):>7,}"
          f"  ({t_prepaid.mean():.2%} of test)")
    print(f"  train Course_Start_Date range : {df['Course_Start_Date'].min().date()}"
          f" .. {df['Course_Start_Date'].max().date()}")
    print(f"  test  Course_Start_Date range : {test_df['Course_Start_Date'].min().date()}"
          f" .. {test_df['Course_Start_Date'].max().date()}")

    scenarios = [
        ("Iteration 0 (Baseline: slice excluded)", ~(prepaid_slice | exclude_unknown)),
        ("Iteration 1 (Reinstated: slice included)", ~exclude_unknown),
    ]

    rows = []
    for name, keep_mask in scenarios:
        df_model = df.loc[keep_mask].copy()
        X, y, dates = build_features(df_model, vocabs)

        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)
        print(f"modeling population: {X.shape[0]:,} rows x {X.shape[1]} features"
              f" | drop rate {y.mean():.4%}")

        # --- Strategy A: random stratified 80/20 holdout (cell 113) ---
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        res_a = evaluate(X_tr, y_tr, X_val, y_val)
        print(f"[A random 80/20]  train n={res_a['n_train']:,}  val n={res_a['n_val']:,}"
              f"  val drop rate={res_a['val_drop_rate']:.4%}")
        print(f"[A random 80/20]  ROC-AUC={res_a['auc']:.4f}"
              f"  FN={res_a['fn']:,}  FP={res_a['fp']:,}  (thr=0.5, fit {res_a['fit_sec']:.0f}s)")

        # --- Strategy B: temporal split, validate on the last N months ---
        cutoff = dates.max() - pd.DateOffset(months=TEMPORAL_VAL_MONTHS)
        tr_mask = dates <= cutoff
        res_b = evaluate(X[tr_mask], y[tr_mask], X[~tr_mask], y[~tr_mask])
        print(f"[B temporal]      cutoff={cutoff.date()}  train n={res_b['n_train']:,}"
              f"  val n={res_b['n_val']:,}  val drop rate={res_b['val_drop_rate']:.4%}")
        print(f"[B temporal]      ROC-AUC={res_b['auc']:.4f}"
              f"  FN={res_b['fn']:,}  FP={res_b['fp']:,}  (thr=0.5, fit {res_b['fit_sec']:.0f}s)")

        rows.append(dict(scenario=name, n_rows=X.shape[0], drop_rate=y.mean(),
                         a=res_a, b=res_b, cutoff=cutoff.date()))

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md = []
    md.append("# PREPAID NONREFUNDABLE slice — impact audit\n")
    md.append(f"Sandbox re-run of the notebook's tuned XGBoost baseline "
              f"(cell 123 best params) under two scenarios. "
              f"The raw `course_start_days` index was replaced with seasonal "
              f"components (`start_month`, `start_dayofweek`) in both scenarios "
              f"to avoid tree extrapolation on the temporal split.\n")
    md.append(f"- Slice definition: `payment_terms_clean == 'PREPAID NONREFUNDABLE'` "
              f"and `enrollment_type_clean != 'AFFILIATED ADMISSION'` "
              f"— **{int(prepaid_slice.sum()):,} train rows**, drop rate "
              f"{df.loc[prepaid_slice, TARGET].mean():.2%}.")
    md.append(f"- Same slice in `Test_Data_No_Target.csv`: **{int(t_prepaid.sum()):,} rows** "
              f"({t_prepaid.mean():.2%} of the {len(test_df):,} test rows).")
    md.append(f"- Temporal validation window: last {TEMPORAL_VAL_MONTHS} months of "
              f"`Course_Start_Date` (cutoff {rows[0]['cutoff']}).\n")
    md.append("| Scenario | Modeling rows | Drop rate | Split | Train n | Val n | "
              "Val drop rate | ROC-AUC | FN @0.5 | FP @0.5 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for tag, res in (("A — random 80/20", r["a"]), ("B — temporal", r["b"])):
            md.append(f"| {r['scenario']} | {r['n_rows']:,} | {r['drop_rate']:.2%} "
                      f"| {tag} | {res['n_train']:,} | {res['n_val']:,} "
                      f"| {res['val_drop_rate']:.2%} | {res['auc']:.4f} "
                      f"| {res['fn']:,} | {res['fp']:,} |")

    a0, b0 = rows[0]["a"], rows[0]["b"]
    a1, b1 = rows[1]["a"], rows[1]["b"]
    md.append("\n## Deltas (Reinstated minus Baseline)\n")
    md.append("| Split | Δ ROC-AUC | Δ FN | Δ FP |")
    md.append("|---|---|---|---|")
    md.append(f"| A — random 80/20 | {a1['auc'] - a0['auc']:+.4f} "
              f"| {a1['fn'] - a0['fn']:+,} | {a1['fp'] - a0['fp']:+,} |")
    md.append(f"| B — temporal | {b1['auc'] - b0['auc']:+.4f} "
              f"| {b1['fn'] - b0['fn']:+,} | {b1['fp'] - b0['fp']:+,} |")
    md.append("\n*Note: the two scenarios are scored on different validation "
              "populations (the reinstated one contains the near-deterministic "
              "slice), so AUC/FN/FP shifts reflect both model change and "
              "population change.*\n")

    with open("prepaid_impact_results.md", "w") as f:
        f.write("\n".join(md))
    print("\nwrote prepaid_impact_results.md")


if __name__ == "__main__":
    main()
