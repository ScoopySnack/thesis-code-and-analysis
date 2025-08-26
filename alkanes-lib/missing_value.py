#!/usr/bin/env python3

# Choose a feature to see which alkanes are missing it
# How to run it:
# 1. Save this script to a file, `missing_value.py`.
# 2. Make sure you have the JSON file (`alkanesStenutz.json`) in the same directory.
# 3. Run the script using Python 3: `python3 missing_value.py`.
# 4. Type in the console the feature to explore (careful what you type to be part of the features in our json).

import json

def get_by_dotpath(d, path):
    """Traverse nested dict using dot-separated path. Returns (exists, value)."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur

def is_missing(value):
    """Define what counts as missing."""
    return value is None or (isinstance(value, str) and value.strip() == "")

def find_alkanes_missing_feature(data, feature_path):
    """Returns (missing_list, available_list) for the given feature path."""
    alkanes = data.get("alkanes", {})
    missing, available = [], []
    for name, feats in alkanes.items():
        exists, val = get_by_dotpath(feats, feature_path)
        if (not exists) or is_missing(val):
            missing.append(name)
        else:
            available.append(name)
    return sorted(missing), sorted(available)

if __name__ == "__main__":
    # Load JSON
    with open("/workspaces/thesis-code-and-analysis/alkanes-lib/alkanesStenutz.json", "r") as f:
        data = json.load(f)

    total = len(data.get("alkanes", {}))
    print(f"Total alkanes: {total}")

    # Ask user which feature to check
    feature = input("Enter feature to check (use dot for nested, e.g. 'refractive_index' or 'critical_point.temperature_Tc'): ").strip()

    # Find missing/available
    missing, available = find_alkanes_missing_feature(data, feature)

    print(f"\n=== Alkanes MISSING feature '{feature}' ===")
    if missing:
        for name in missing:
            print(f"- {name}")
    else:
        print("(none)")
    print(f"Missing: {len(missing)} / {total} ({(len(missing)/total*100 if total else 0):.1f}%)")

    print(f"\n=== Alkanes WITH feature '{feature}' ===")
    if available:
        for name in available:
            print(f"- {name}")
    else:
        print("(none)")
    print(f"Available: {len(available)} / {total} ({(len(available)/total*100 if total else 0):.1f}%)")
