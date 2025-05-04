import requests


# Function to fetch multiple properties from PubChem using SMILES
def get_properties_from_pubchem(smiles):
    # PubChem PUG REST API URL for fetching multiple properties based on SMILES
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/MolecularWeight,MeltingPoint,BoilingPoint,Density,ExactMass,CanonicalSMILES,json"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        try:
            properties = data['PropertyTable']['Properties'][0]
            return properties
        except KeyError:
            return "Properties not available"
    else:
        return "Error fetching data from PubChem"


# Example: Fetching properties for Butane (C4H10)
smiles = "CCCC"

properties = get_properties_from_pubchem(smiles)

# Print the fetched properties
if isinstance(properties, dict):
    print(f"Properties for {smiles}:")
    for key, value in properties.items():
        print(f"{key}: {value}")
else:
    print(properties)
