# -*- coding: utf-8 -*-
# stenutz_c1_c10_isomers_to_json.py  (drop-in for alkanes-lib/stenutz_fetch_test.py)

from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

BASE = "https://www.stenutz.eu/chem/"
SOLV = urljoin(BASE, "solv6.php")           # e.g. solv6.php?name=octane
ALT  = urljoin(BASE, "solv6%20(2).php")     # some isomers live here too
TIMEOUT = 20

N_ALKANES = [
    "methane", "ethane", "propane", "butane", "pentane",
    "hexane", "heptane", "octane", "nonane", "decane"
]
CARBON_MIN, CARBON_MAX = 1, 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Python-requests/Codespaces"
}

# ---------------------- helpers ----------------------

def _get(url: str, params=None) -> BeautifulSoup:
    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")  # works with built-in or html5lib if installed

def _num(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None

def _parse_formula(s: str) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    m = re.search(r"C\s*(\d+)\s*H\s*(\d+)", s.replace(" ", ""), re.I)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _is_alkane(formula: str) -> bool:
    p = _parse_formula(formula)
    if not p:
        return False
    c, h = p
    return h == 2 * c + 2

def _mw_from_CH(formula: str) -> Optional[float]:
    p = _parse_formula(formula)
    if not p:
        return None
    c, h = p
    return round(c * 12.011 + h * 1.008, 2)

def _page_text(soup: BeautifulSoup) -> str:
    # single-spaced text; easier regex across weird punctuation
    return soup.get_text(" ", strip=True)

def _grab_after_label(text: str, label_regex: str) -> Optional[str]:
    """
    Find the value that follows a label like 'Label: value'.
    Returns None if not found. Never raises AttributeError.
    """
    patterns = [
        label_regex + r"\s*[:\-–]\s*([^\n]+?)\s{2,}",    # up to big gap
        label_regex + r"\s*[:\-–]\s*([^\n]+)",          # to end of line
        label_regex + r"\s+([^\n]+?)\s{2,}",            # label value (no colon)
        label_regex + r"\s+([^\n]+)",                   # last resort
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m and m.group(1):
            return m.group(1).strip()
    return None

def _parse_formula_and_mw(field: str) -> tuple[Optional[str], Optional[float]]:
    """
    Accepts fields like 'C2H6; 30.07 g/mol' or 'C2H6 30.07 g·mol−1'
    Returns ('C2H6', 30.07)
    """
    if not field:
        return None, None
    no_spaces = field.replace(" ", "")
    f = re.search(r"C\d+H\d+", no_spaces, re.I)
    formula = f.group(0) if f else None

    mm = None
    with_units = re.search(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:g(?:/|·)?mol(?:-1)?)", field.replace(",", "."),
        re.I
    )
    if with_units:
        mm = float(with_units.group(1))
    else:
        post = re.search(r"C\d+H\d+\D*([0-9]+(?:\.[0-9]+)?)", field.replace(",", "."), re.I)
        if post:
            mm = float(post.group(1))
        else:
            last = re.findall(r"[0-9]+(?:\.[0-9]+)?", field.replace(",", "."))
            if last:
                try:
                    mm = float(last[-1])
                except Exception:
                    mm = None
    return formula, mm

def _parse_compound_page(url: str) -> Optional[Dict[str, Any]]:
    soup = _get(url)
    txt = _page_text(soup)

    title = soup.find(["h1", "h2"])
    name = title.get_text(strip=True) if title else None

    formula_field = _grab_after_label(txt, r"\bFormula\b")
    if not formula_field:
        maybe = re.search(r"C\s*\d+\s*H\s*\d+", txt, re.I)
        formula_field = maybe.group(0) if maybe else None

    formula, mw_from_page = _parse_formula_and_mw(formula_field or "")
    if not formula or not _is_alkane(formula):
        return None

    parsed = _parse_formula(formula)
    if not parsed:
        return None
    c_atoms, _ = parsed
    if not (CARBON_MIN <= c_atoms <= CARBON_MAX):
        return None

    dens = _grab_after_label(txt, r"\bDensity\b")
    nD   = _grab_after_label(txt, r"\bRefractive\s+index\b|\bnD\b")
    MR   = _grab_after_label(txt, r"\bMolecular\s+refractive\s+power\b|\bMolecular\s+refraction\b|\bMR\b")
    mp   = _grab_after_label(txt, r"\bMelting\s+point\b|\bmp\b")
    bp   = _grab_after_label(txt, r"\bBoiling\s+point\b|\bbp\b")
    mv   = _grab_after_label(txt, r"\bMolar\s+volume\b")

    Tc   = _grab_after_label(txt, r"\bCritical\s+temperature\b|\bT[cC]\b")
    Pc   = _grab_after_label(txt, r"\bCritical\s+pressure\b|\bP[cC]\b")
    Vc   = _grab_after_label(txt, r"\bCritical\s+volume\b|\bV[cC]\b")

    mw_final = mw_from_page if mw_from_page is not None else _mw_from_CH(formula)

    return {
        "key": name or url,
        "number_ofC": c_atoms,
        "molecular_weight": mw_final,
        "Density": _num(dens),
        "molar_volume": _num(mv),
        "refractive_index": _num(nD),
        "Molecular_refractive_power": _num(MR),
        "dielectric_constant": None,
        "dipole_moment": 0.0,
        "melting_point": _num(mp),
        "boiling_point": _num(bp),
        "vapour_pressure": None,
        "surface_tension": None,
        "viscosity": None,
        "critical_point": {
            "temperature_Tc": _num(Tc),
            "pressure_Pc": _num(Pc),
            "volume_Vc": _num(Vc),
        },
        "logP": None,
        "δ": None,
        "specific_heat_capacity": None,
    }

def _isomer_links_for(alkane: str) -> List[Tuple[str, str]]:
    """
    Collect links to isomer pages from solv6.php?name=<alkane>.
    Scans all <a> with 'solv6' in href and a 'name=' query param.
    Always includes the base alkane page.
    """
    soup = _get(SOLV, params={"name": alkane})
    links: List[Tuple[str, str]] = [(alkane, f"{SOLV}?name={alkane}")]
    for a in soup.select('a[href*="solv6"]'):
        href = a.get("href", "")
        full = urljoin(BASE, href)
        q = parse_qs(urlparse(full).query)
        nm = (q.get("name", [a.get_text(strip=True)])[0] or "").strip()
        if not nm:
            continue
        if all(nm != n for (n, _) in links):
            links.append((nm, full))
    return links

# ---------------------- main build ----------------------

def build_c1_c10_alkanes() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for alk in N_ALKANES:
        print(f"[info] Crawling {alk} …")
        isomer_links = _isomer_links_for(alk)
        for nm, href in isomer_links:
            try:
                data = _parse_compound_page(href)
            except Exception as e:
                print(f"[warn] skip {nm} @ {href} -> {e}")
                continue
            if not data:
                print(f"[warn] no data for {nm} @ {href}")
                continue
            key = nm  # keep IUPAC/common names as keys
            result[key] = {
                "number_ofC": data["number_ofC"],
                "molecular_weight": data["molecular_weight"],
                "Density": data["Density"],
                "molar_volume": data["molar_volume"],
                "refractive_index": data["refractive_index"],
                "Molecular_refractive_power": data["Molecular_refractive_power"],
                "dielectric_constant": data["dielectric_constant"],
                "dipole_moment": data["dipole_moment"],
                "melting_point": data["melting_point"],
                "boiling_point": data["boiling_point"],
                "vapour_pressure": data["vapour_pressure"],
                "surface_tension": data["surface_tension"],
                "viscosity": data["viscosity"],
                "critical_point": data["critical_point"],
                "logP": data["logP"],
                "δ": data["δ"],
                "specific_heat_capacity": data["specific_heat_capacity"],
            }

    # Sort by carbon count then name; wrap at root
    result = dict(sorted(result.items(), key=lambda kv: ((kv[1]["number_ofC"] or 999), kv[0].lower())))
    return {"alkanes": result}

if __name__ == "__main__":
    data = build_c1_c10_alkanes()
    out_path = "alkanes_C1_C10_isomers_stenutz.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[done] Saved {len(data['alkanes'])} compounds to {out_path}")
