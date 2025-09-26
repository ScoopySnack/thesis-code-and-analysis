import json
import pandas as pd
from collections import defaultdict

# ---------- Load JSON ----------
with open("/workspaces/thesis-code-and-analysis/alkanes-lib/alkanes_top_features_0p9_1p0.json", "r") as f:
    data = json.load(f)

# ---------- Feature availability across all alkanes ----------
stats = defaultdict(lambda: {"missing": 0, "total": 0})
for _, features in data["alkanes"].items():
    for key, value in features.items():
        if isinstance(value, dict):
            for subk, subv in value.items():
                fname = f"{key}.{subk}"
                stats[fname]["total"] += 1
                if subv in ("", None, " "):
                    stats[fname]["missing"] += 1
        else:
            stats[key]["total"] += 1
            if value in ("", None, " "):
                stats[key]["missing"] += 1

df_features = pd.DataFrame.from_dict(stats, orient="index")
df_features["available"] = df_features["total"] - df_features["missing"]
df_features["availability_rate"] = df_features["available"] / df_features["total"]

# ---------- Select features by threshold ----------
threshold_min, threshold_max = 0.9, 1.0
selected_features = df_features.loc[
    (df_features["availability_rate"] >= threshold_min) &
    (df_features["availability_rate"] <= threshold_max)
].index.tolist()

# ---------- Extract alkanes with all selected features ----------
filtered_alkanes = {}
for name, feats in data["alkanes"].items():
    keep = True
    for feat in selected_features:
        if "." in feat:  # nested feature
            k, sub = feat.split(".", 1)
            if not isinstance(feats.get(k), dict) or feats[k].get(sub) in ("", None, " "):
                keep = False
                break
        else:
            if feats.get(feat) in ("", None, " "):
                keep = False
                break
    if keep:
        filtered_alkanes[name] = feats

# ---------- Save new JSON ----------
out_path = "/workspaces/thesis-code-and-analysis/alkanes-lib/alkanes_top_features_0p9_1p0_1.json"
with open(out_path, "w") as f:
    json.dump({"alkanes": filtered_alkanes}, f, indent=2)

print(f"Selected features: {selected_features}")
print(f"Extracted {len(filtered_alkanes)} alkanes -> {out_path}")
