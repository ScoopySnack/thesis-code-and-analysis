import json

# Load JSON file with correct path
with open("/workspaces/thesis-code-and-analysis/alkanes-lib/alkanesStenutz.json", "r") as f:
    data = json.load(f)

# Count alkanes
count = len(data.get("alkanes", {}))

# Print result
print(f"Number of alkanes in JSON: {count}")
