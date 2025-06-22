import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("alkanes_final.csv")

# Drop graph-based columns
graph_cols = ["perron_frobenius", "fiedler_eigenvalue", "compression_ratio", "information_content"]
df = df.drop(columns=graph_cols, errors="ignore")

# Define targets to predict
targets_to_predict = [
    "viscosity",
    "d",
    "specific_heat_capacity"
]

for target in targets_to_predict:
    print(f"\n🔍 Processing target: {target}")

    # Only proceed if the target exists
    if target not in df.columns:
        print(f"⚠️ Skipping {target} (not in dataset)")
        continue

    # Split into known and unknown
    known = df[df[target].notna()]
    unknown = df[df[target].isna()]

    # Skip if all values are already known or missing
    if len(known) == 0 or len(unknown) == 0:
        print(f"⚠️ Skipping {target} (nothing to predict)")
        continue

    # Use all other numeric features except the target itself
    feature_cols = known.select_dtypes(include=["float64", "int64"]).columns.drop([target])
    X_known = known[feature_cols]
    y_known = known[target]
    X_unknown = unknown[feature_cols]

    # Train and predict
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_known, y_known)
    y_pred = model.predict(X_unknown)

    # Fill predictions back into DataFrame
    df.loc[df[target].isna(), target] = y_pred
    print(f"✅ Filled {len(y_pred)} missing values in '{target}'")

# Save final enriched dataset
df.to_csv("alkanes_predicted.csv", index=False)
print("\n✅ All done! Saved as 'alkanes_predicted.csv'")
