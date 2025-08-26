import json
import pandas as pd
from collections import defaultdict

# ---------- Load JSON ----------
with open("/workspaces/thesis-code-and-analysis/alkanes-lib/alkanesStenutz.json", "r") as f:
    data = json.load(f)

# Count alkanes
count = len(data.get("alkanes", {}))
print(f"Number of alkanes in JSON: {count}")


# Script to measure feature availability in order to see which features are 
# most commonly missing and which alkanes are most affected by these missing features. 
# For ML this can be useful for feature selection and understanding data quality.

# ---------- Per-alkane missing/available counts ----------
def per_alkane_missing_counts(data):
    counts = {}
    for alkane, features in data["alkanes"].items():
        missing = 0
        total = 0
        for key, value in features.items():
            if isinstance(value, dict):  # nested dict (e.g., critical_point)
                for subk, subv in value.items():
                    total += 1
                    if subv in ("", None):
                        missing += 1
            else:
                total += 1
                if value in ("", None):
                    missing += 1
        counts[alkane] = {"missing": missing, "total": total, "available": total - missing}
    return pd.DataFrame.from_dict(counts, orient="index").sort_values("missing")

# ---------- Feature availability across all alkanes ----------
def feature_availability(data):
    stats = defaultdict(lambda: {"missing": 0, "total": 0})
    for _, features in data["alkanes"].items():
        for key, value in features.items():
            if isinstance(value, dict):
                for subk, subv in value.items():
                    fname = f"{key}.{subk}"
                    stats[fname]["total"] += 1
                    if subv in ("", None):
                        stats[fname]["missing"] += 1
            else:
                stats[key]["total"] += 1
                if value in ("", None):
                    stats[key]["missing"] += 1

    df = pd.DataFrame.from_dict(stats, orient="index")
    df["available"] = df["total"] - df["missing"]
    df["availability_rate"] = df["available"] / df["total"]
    return df.sort_values(["availability_rate", "available"], ascending=[False, False])

# ---------- Main ----------
if __name__ == "__main__":
    # Per-alkane analysis
    df_alkanes = per_alkane_missing_counts(data)
    print("\n=== Per-alkane missing/available counts ===")
    print(df_alkanes)

    # Feature-level analysis
    df_features = feature_availability(data)
    print("\n=== Feature availability (best to worst) ===")
    print(df_features)

    print("\n=== Most incomplete features (worst 10) ===")
    worst = df_features.sort_values(["availability_rate", "missing"], ascending=[True, False]).head(10)
    print(worst[["missing", "total", "available", "availability_rate"]])
