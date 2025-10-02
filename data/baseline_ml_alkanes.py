import os
# Limit joblib/loky worker detection on Windows before sklearn imports
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# ========== CONFIG ==========
# Resolve data path robustly whether the script is run from repo root or the data folder
_THIS_FILE = Path(__file__).resolve()
_BASE_DIR = _THIS_FILE.parent
_DATA_CANDIDATES = [
    _BASE_DIR / "alkanes_core_with_smiles_final.csv",
    _BASE_DIR.parent / "data" / "alkanes_core_with_smiles_final.csv",
    Path("data") / "alkanes_core_with_smiles_final.csv",
    Path("alkanes_core_with_smiles_final.csv"),
]
for _p in _DATA_CANDIDATES:
    if _p.exists():
        DATA_PATH = _p
        break
else:
    raise FileNotFoundError(
        "Could not find 'alkanes_core_with_smiles_final.csv'. Tried: " +
        ", ".join(str(p) for p in _DATA_CANDIDATES)
    )

TARGETS = ["boiling_point", "Density"]  # you can add "melting_point"
RANDOM_STATE = 42
N_FOLDS = 5
PARALLEL_NJOBS = 1   # keep 1 on Windows
# ============================

df = pd.read_csv(DATA_PATH)

id_like = {"name", "formula", "SMILES"}
num_cols = [c for c in df.columns if c not in id_like]
# Keep only numeric columns and drop columns that are entirely NaN (which break imputers/estimators)
num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
num_cols = [c for c in num_cols if df[c].notna().any()]
X_all = df[num_cols].copy()

def evaluate_one_target(target: str):
    if target not in X_all.columns:
        print(f"[SKIP] Target '{target}' not found.")
        return

    y = X_all[target].copy()
    X = X_all.drop(columns=[target])

    # Drop rows where the target is missing; estimators cannot handle NaN targets
    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), X.columns)
    ])

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=1
        ),
    }

    print(f"\n===== TARGET: {target} =====")
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])

        # ⚠️ Use neg_mean_squared_error and take sqrt for RMSE
        cv = cross_validate(
            pipe, X_train, y_train,
            cv=N_FOLDS,
            scoring=("r2", "neg_mean_absolute_error", "neg_mean_squared_error"),
            n_jobs=PARALLEL_NJOBS,
            return_train_score=False
        )

        r2_cv = cv["test_r2"].mean()
        mae_cv = -cv["test_neg_mean_absolute_error"].mean()
        rmse_cv = (-cv["test_neg_mean_squared_error"].mean()) ** 0.5  # √MSE

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        # Older sklearn may not support 'squared' param; compute RMSE via sqrt(MSE)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5  # RMSE

        print(f"\n[{name}] CV: R²={r2_cv:.3f} | MAE={mae_cv:.3f} | RMSE={rmse_cv:.3f}")
        print(f"[{name}]  T: R²={r2:.3f}  | MAE={mae:.3f}  | RMSE={rmse:.3f}")

        if name == "RandomForest":
            pre_fit = pre.fit(X_train)
            X_train_pre = pre_fit.transform(X_train)
            rf = RandomForestRegressor(
                n_estimators=400, random_state=RANDOM_STATE, n_jobs=1
            ).fit(X_train_pre, y_train)
            importances = rf.feature_importances_
            feats = list(X.columns)
            top = sorted(zip(feats, importances), key=lambda t: t[1], reverse=True)[:8]
            print("   Top features:", ", ".join([f"{f} ({w:.3f})" for f, w in top]))

for t in TARGETS:
    evaluate_one_target(t)
