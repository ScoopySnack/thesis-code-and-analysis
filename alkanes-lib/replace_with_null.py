import json

def replace_empty_with_null(obj):
    """Recursively replace empty strings with None in a JSON object."""
    if isinstance(obj, dict):
        return {k: replace_empty_with_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_empty_with_null(v) for v in obj]
    elif obj == "":
        return None
    else:
        return obj

# Path to your JSON file
json_file = "/workspaces/thesis-code-and-analysis/alkanes-lib/alkanesStenutz.json"

# Load, modify, and overwrite in place
with open(json_file, "r+", encoding="utf-8") as f:
    data = json.load(f)
    data = replace_empty_with_null(data)

    # reset file pointer & overwrite the same file
    f.seek(0)
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.truncate()  # remove leftover content if new JSON is shorter
