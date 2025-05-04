from rdkit import Chem
from rdkit.Chem import Descriptors
import requests

physical_state = {'liquid', 'solid', 'gas'}

# def molecule(name, formula, state):
#     mol = Chem.MolFromSmiles(formula)
#     mol.name = name
#     mol.physical_state = state
#     return mol
#
# # Example usage
# print(molecule('Water', 'O', 'liquid'))

# Fetching melting point from PubChem
def get_melting_point(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/MeltingPoint/TXT"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text.strip()
    else:
        return "Melting point not available"


methane = Chem.MolFromSmiles('C')
methane.physical_state = 'gas'
#melting point
methane.melting_point = get_melting_point('C')
print(f"Melting point: {methane.melting_point}")

# boiling point
def get_boiling_point(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/BoilingPoint/TXT"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text.strip()
    else:
        return "Boiling point not available"

methane.boiling_point = get_boiling_point('C')
print(f"Boiling point: {methane.boiling_point}")

