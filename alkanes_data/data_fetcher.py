from chemicals.critical import Tc, Pc, Vc
from chemicals.viscosity import mu_l
from chemicals.surfacetension import sigma
from chemicals.identifiers import lookup_CAS_from_any
from chemicals.utils import molecular_weight
from rdkit.Chem import Descriptors
from rdkit import Chem

def fetch_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        cas = lookup_CAS_from_any(smiles)
        props = {
            "molar_mass": molecular_weight(cas),
            "Tc": Tc(cas),
            "Pc": Pc(cas),
            "Vc": Vc(cas),
            "viscosity_25C": mu_l(cas, 298.15),
            "surface_tension_25C": sigma(cas, 298.15)
        }
    except:
        props = {
            "molar_mass": Descriptors.MolWt(mol),
            "Tc": None,
            "Pc": None,
            "Vc": None,
            "viscosity_25C": None,
            "surface_tension_25C": None
        }
    return props