# Improve parser: handle di/tri/tetra prefixes and add 'nonane' base length.
import re
import json
from pathlib import Path

src = Path("/mnt/data/alkanesStenutz.updated.smiles.auto.json")
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

alkanes = data["alkanes"]

BASE_LEN = {
    "butane": 4,
    "pentane": 5,
    "hexane": 6,
    "heptane": 7,
    "octane": 8,
    "nonane": 9,
    "decane": 10,
}

def base_from_name(name: str):
    for base in BASE_LEN.keys():
        if name.endswith(base):
            return base
    return None

def build_linear_chain(n: int):
    return ["C"] * n

def insert_branch(smiles_list, position: int, branch: str):
    idx = position - 1
    if 0 <= idx < len(smiles_list):
        smiles_list[idx] = smiles_list[idx] + branch

def generate_smiles_from_pattern(name: str) -> str:
    base = base_from_name(name)
    if not base:
        return ""
    n = BASE_LEN[base]
    chain = build_linear_chain(n)

    prefix = name[: -len(base)]
    pattern = re.compile(r'(\d+(?:,\d+)*)-(?:di|tri|tetra)?(methyl|ethyl)', re.IGNORECASE)
    subs = []
    for match in pattern.finditer(prefix):
        positions_str, sub = match.groups()
        positions = [int(p) for p in positions_str.split(",")]
        subs.append({"sub": sub.lower(), "positions": positions})
    if not subs:
        return ""

    for seg in subs:
        branch = "(C)" if seg["sub"] == "methyl" else "(CC)"
        for pos in seg["positions"]:
            insert_branch(chain, pos, branch)

    return "".join(chain)

auto_updated = []
auto_failed = []
for name, props in alkanes.items():
    if props.get("SMILES"):
        continue
    s = generate_smiles_from_pattern(name)
    if s:
        props["SMILES"] = s
        auto_updated.append(name)
    else:
        auto_failed.append(name)

out2 = Path("/mnt/data/alkanesStenutz.updated.smiles.auto2.json")
with open(out2, "w", encoding="utf-8") as f:
    json.dump({"alkanes": alkanes}, f, indent=2, ensure_ascii=False)

len(auto_updated), len(auto_failed), auto_failed[:25], str(out2)
