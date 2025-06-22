import json

# Step 1 – Load original file
with open("alkanesStenutz.json", "r") as file:
    data = json.load(file)

def normalize_key(key):
    """Convert keys to lowercase and use underscores."""
    return key.strip().lower().replace(" ", "_")

def clean_and_normalize(obj):
    """Recursive function to normalize keys and replace empty strings."""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            new_key = normalize_key(key)
            cleaned[new_key] = clean_and_normalize(value)
        return cleaned
    elif isinstance(obj, list):
        return [clean_and_normalize(v) for v in obj]
    elif obj == "":
        return None
    else:
        return obj

# Step 2 – Clean and normalize everything
cleaned_data = clean_and_normalize(data)

# Step 3 – Add placeholder graph_properties to each alkane
for alkane in cleaned_data.get("alkanes", {}):
    cleaned_data["alkanes"][alkane]["graph_properties"] = {
        "perron_frobenius": None,
        "fiedler_eigenvalue": None,
        "compression_ratio": None,
        "information_content": None
    }

# Step 4 – Save to new file
with open("alkanes_cleaned.json", "w") as outfile:
    json.dump(cleaned_data, outfile, indent=2)

print("✅ Done! Saved as 'alkanes_cleaned.json'")
