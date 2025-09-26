import json
import csv

# Load the enriched JSON file
with open("/workspaces/thesis-code-and-analysis/alkanes-lib/alkanes_top_features_0p9_1p0_1.json", "r") as file:
    data = json.load(file)

# Extract all alkanes
alkanes = data["alkanes"]

# Flatten each alkane's properties
flat_data = []

for name, props in alkanes.items():
    row = {
        "name": name,
        "smiles": props.get("smiles"),
        "number_ofC": props.get("number_ofc"),
        "molecular_weight": props.get("molecular_weight"),
        "density": props.get("density"),
        "molar_volume": props.get("molar_volume"),
        "refractive_index": props.get("refractive_index"),
        "dielectric_constant": props.get("dielectric_constant"),
        "dipole_moment": props.get("dipole_moment"),
        "melting_point": props.get("melting_point"),
        "boiling_point": props.get("boiling_point"),
        "vapour_pressure": props.get("vapour_pressure"),
        "surface_tension": props.get("surface_tension"),
        "viscosity": props.get("viscosity"),
        "logP": props.get("logp"),
        "δ": props.get("δ"),
        "specific_heat_capacity": props.get("specific_heat_capacity")
    }

    # Add critical point fields
    crit = props.get("critical_point", {})
    row["Tc"] = crit.get("temperature_tc")
    row["Pc"] = crit.get("pressure_pc")
    row["Vc"] = crit.get("volume_vc")

    # Add graph properties
    graph = props.get("graph_properties", {})
    row["perron_frobenius"] = graph.get("perron_frobenius")
    row["fiedler_eigenvalue"] = graph.get("fiedler_eigenvalue")
    row["compression_ratio"] = graph.get("compression_ratio")
    row["information_content"] = graph.get("information_content")

    flat_data.append(row)

# Get all column headers
fieldnames = list(flat_data[0].keys())

# Write to CSV
with open("alkanes_final.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flat_data)

print("✅ CSV created as 'alkanes_final.csv'")
