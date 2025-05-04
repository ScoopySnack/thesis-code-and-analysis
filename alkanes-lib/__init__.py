import json
import csv

with open("LinearAlkanes.json", "r") as f:
    data = json.load(f)

print(data)

def open_json_file(file_path):
    with open(file_path, "r") as f:
        data1 = json.load(f)
    return data1

def json_to_dict(json_data):
    return json.loads(json_data)


def json_to_csv(json_filename, csv_filename):
    """Converts a JSON file to a CSV file."""
    # Open and load the JSON file
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)

    # Open the CSV file for writing
    with open(csv_filename, 'w', newline='') as csv_file:
        if not data:
            print("No data in the JSON file to write to CSV.")
            return

        # Create a CSV DictWriter object using the keys from the first item in the JSON list
        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())

        # Write the header (column names)
        writer.writeheader()

        # Write the rows (records)
        writer.writerows(data)

    print(f"Data has been written to {csv_filename}")


# Example usage
json_filename = "LinearAlkanes.json"
csv_filename = "LinearAlkanes.csv"
json_to_csv(json_filename, csv_filename)


