#!/usr/bin/env python
"""
prompt_blocks.py — shared helpers for prompt-side molecular context injection.

Three text blocks can be injected into the LLM prediction prompt:
  1. synonyms_text(smiles, syn_cache)      — compound name / IUPAC / aliases
  2. functional_groups_text(smiles)        — AccFG functional group list
  3. rdkit_descriptors_text(smiles, ...)   — task-relevant RDKit descriptors

Each returns an empty string if the relevant data isn't available, so the
caller can safely concatenate.

Used by:
  - chemrag_retrieval.py    (retrieval + prompt injection)
  - mol_context_only.py   (prompt injection only, no retrieval)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent  # src/ -> MolE-RAG/
RDKIT_FEATURE_CACHE_PATH = REPO_ROOT / "caches" / "task_rdkit_features.json"
DEFAULT_SYNONYM_CACHE     = REPO_ROOT / "caches" / "llm_filtered_synonyms.json"


# ===========================================================================
# Canonical-SMILES lookup helper
# ===========================================================================
def _canon_for_lookup(smiles: str) -> str:
    """RDKit canonical SMILES, for synonym cache lookups.

    Cache keys were re-keyed to canonical SMILES by canonicalize_synonyms.py.
    This canonicalizes the query SMILES so lookups succeed regardless of how
    the SMILES was originally written in the test CSV.
    Returns the input unchanged if RDKit is unavailable or parsing fails.
    """
    if not smiles:
        return smiles
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else smiles
    except Exception:
        return smiles


# ===========================================================================
# 1. Synonyms / IUPAC / compound identifiers
# ===========================================================================
def load_syn_cache(path: Path = DEFAULT_SYNONYM_CACHE) -> Optional[Dict[str, Any]]:
    """Load the LLM-filtered synonym cache. Returns None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def synonyms_text(smiles: str, syn_cache: Optional[Dict[str, Any]]) -> str:
    """Return a text block with the molecule's name, IUPAC, and top synonyms.
    Empty string if no entry exists for this SMILES."""
    if syn_cache is None:
        return ""
    canon_smiles = _canon_for_lookup(smiles)
    entry = syn_cache.get(canon_smiles)
    if not isinstance(entry, dict):
        return ""
    iupac = (entry.get("iupac_name") or "").strip()
    good_syns = entry.get("good_syns") or []
    if not (iupac or good_syns):
        return ""
    lines = ["Compound identifiers:"]
    if good_syns:
        # First synonym is usually the most common / canonical name
        lines.append(f"  Name: {good_syns[0]}")
        if len(good_syns) > 1:
            other_names = ", ".join(good_syns[1:6])  # up to 5 additional
            lines.append(f"  Other names: {other_names}")
    if iupac:
        lines.append(f"  IUPAC: {iupac}")
    return "\n".join(lines)


# ===========================================================================
# 2. AccFG functional groups
# ===========================================================================
try:
    from accfg import AccFG as _AccFG_class  # noqa: E402
    _ACCFG_AVAILABLE = True
except ImportError:
    _ACCFG_AVAILABLE = False
    _AccFG_class = None  # type: ignore

_accfg_instance = None


def _get_accfg():
    """Lazy-init AccFG. Uses FULL mode (lite=False) for fine-grained FGs."""
    global _accfg_instance
    if _accfg_instance is None and _ACCFG_AVAILABLE:
        _accfg_instance = _AccFG_class(lite=False, print_load_info=False)
    return _accfg_instance


def get_functional_groups(smiles: str) -> List[str]:
    """Return list of functional group names present in the molecule.
    Empty list if AccFG isn't available or extraction failed.

    AccFG's return type varies with flags and version:
      - default flags → list of FG names
      - show_atoms=True → dict {fg_name: [atom_indices]}
      - show_graph=True → tuple of (fgs, graph)
    We handle all of these.
    """
    if not _ACCFG_AVAILABLE:
        return []
    afg = _get_accfg()
    if afg is None:
        return []
    try:
        result = afg.run(smiles)
    except Exception:
        return []
    # Unwrap tuple if present
    if isinstance(result, tuple):
        result = result[0]
    # Convert to list of FG name strings
    if isinstance(result, dict):
        return list(result.keys())
    if isinstance(result, list):
        return [str(x) for x in result if x]
    return []


def functional_groups_text(smiles: str) -> str:
    """Return a one-line text block listing the molecule's functional groups.
    Empty string if no FGs detected or AccFG unavailable."""
    fgs = get_functional_groups(smiles)
    if not fgs:
        return ""
    return ("This molecule has the following functional groups: "
            + ", ".join(fgs) + ".")


# ===========================================================================
# 3. RDKit descriptors (task-relevant, ranked by correlation/SHAP on val)
# ===========================================================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _bucket(v: float, lo: float, hi: float, labels: Tuple[str, str, str]) -> str:
    """Bucket value v into one of 3 qualitative labels."""
    if v < lo:  return labels[0]
    if v < hi:  return labels[1]
    return labels[2]


# Qualitative label thresholds for commonly used descriptors. If a descriptor
# is in this map, we append a qualitative label after the numeric value.
# Thresholds drawn from typical chemistry vocabulary in the literature.
_DESCRIPTOR_LABELS: Dict[str, Tuple[float, float, Tuple[str, str, str]]] = {
    "MolWt":      (300, 500, ("low molecular weight",
                              "moderate molecular weight",
                              "high molecular weight")),
    "MolLogP":    (1.0, 3.0, ("hydrophilic",
                              "moderate lipophilicity",
                              "highly lipophilic")),
    "TPSA":       (60.0, 120.0, ("low polar surface area",
                                 "moderate polar surface area",
                                 "high polar surface area")),
    "qed":        (0.4, 0.7, ("low drug-likeness",
                              "moderate drug-likeness",
                              "high drug-likeness")),
    "NumHDonors":      (1.5, 4.5, ("few H-bond donors",
                                   "moderate H-bond donors",
                                   "many H-bond donors")),
    "NumHAcceptors":   (3.5, 7.5, ("few H-bond acceptors",
                                   "moderate H-bond acceptors",
                                   "many H-bond acceptors")),
    "NumRotatableBonds": (3.5, 7.5, ("few rotatable bonds",
                                     "moderate rotatable bonds",
                                     "many rotatable bonds")),
}


# Inline descriptor explanations. Each descriptor that appears in a prompt
# gets a one-line plain-English description so the LLM knows what the value
# represents. Covers commonly surfaced RDKit descriptors in MoleculeNet
# correlation analyses.
_DESCRIPTOR_DESCRIPTIONS: Dict[str, str] = {
    # Basic physicochemical
    "MolWt":           "molecular weight in Daltons",
    "ExactMolWt":      "exact monoisotopic molecular weight",
    "HeavyAtomMolWt":  "molecular weight excluding hydrogens",
    "MolLogP":         "calculated octanol-water partition coefficient (lipophilicity)",
    "MolMR":           "molar refractivity (polarizability and volume)",
    "TPSA":            "topological polar surface area; high values reduce membrane permeability",
    "qed":             "quantitative estimate of drug-likeness (0=poor, 1=excellent)",
    "NumHDonors":      "number of hydrogen-bond donor groups (N-H, O-H)",
    "NumHAcceptors":   "number of hydrogen-bond acceptor groups (N, O lone pairs)",
    "NumRotatableBonds":   "number of freely rotatable single bonds (flexibility proxy)",
    "NumAromaticRings":    "number of aromatic ring systems",
    "NumAliphaticRings":   "number of non-aromatic ring systems",
    "NumSaturatedRings":   "number of fully saturated (sp3-only) ring systems",
    "HeavyAtomCount":  "number of non-hydrogen atoms",
    "RingCount":       "total number of ring systems",
    "FractionCSP3":    "fraction of sp3 hybridized carbons (3D character)",

    # Topological complexity
    "BertzCT":         "molecular complexity index; higher = more complex",
    "BalabanJ":        "topological connectivity index; higher = more branched",
    "Ipc":             "information content of molecular graph",
    "HallKierAlpha":   "Hall-Kier alpha shape parameter (negative = more drug-like)",

    # Chi connectivity indices
    "Chi0":            "atomic connectivity index, order 0 (size proxy)",
    "Chi0n":           "size-normalized Chi0",
    "Chi0v":           "valence-corrected Chi0",
    "Chi1":            "atomic connectivity index, order 1",
    "Chi1n":           "size-normalized Chi1",
    "Chi1v":           "valence-corrected Chi1",
    "Chi2n":           "connectivity index, order 2 (branching pattern)",
    "Chi2v":           "valence Chi2",
    "Chi3n":           "connectivity index, order 3",
    "Chi3v":           "valence Chi3",
    "Chi4n":           "connectivity index, order 4",
    "Chi4v":           "valence Chi4",

    # Kappa shape
    "Kappa1":          "Kappa shape index (linearity)",
    "Kappa2":          "Kappa shape index (branching)",
    "Kappa3":          "Kappa shape index (centrality)",

    # Partial charges
    "MaxPartialCharge":    "largest positive partial atomic charge",
    "MinPartialCharge":    "largest negative partial atomic charge",
    "MaxAbsPartialCharge": "largest absolute partial charge magnitude",
    "MinAbsPartialCharge": "smallest absolute partial charge magnitude",

    # E-state indices
    "MaxAbsEStateIndex":   "largest absolute electrotopological state index",
    "MaxEStateIndex":      "largest signed electrotopological state index",
    "MinAbsEStateIndex":   "smallest absolute electrotopological state index",
    "MinEStateIndex":      "smallest signed electrotopological state index",

    # Atom and group counts
    "NOCount":             "count of nitrogen and oxygen atoms (polar atoms)",
    "NHOHCount":           "count of N-H and O-H groups (H-bond donor atoms)",
    "NumHeteroatoms":      "count of non-carbon non-hydrogen atoms",
    "NumValenceElectrons": "total valence electron count",
    "NumRadicalElectrons": "unpaired electron count",
    "NumAromaticCarbocycles":   "number of aromatic all-carbon rings",
    "NumAromaticHeterocycles":  "number of aromatic heterocyclic rings",
    "NumSaturatedCarbocycles":  "number of saturated all-carbon rings",
    "NumSaturatedHeterocycles": "number of saturated heterocyclic rings",
    "NumAliphaticCarbocycles":  "number of aliphatic all-carbon rings",
    "NumAliphaticHeterocycles": "number of aliphatic heterocyclic rings",

    # Surface area / volume
    "LabuteASA":           "approximate molecular surface area (Angstroms^2)",

    # Morgan fingerprint density
    "FpDensityMorgan1":    "Morgan fingerprint density at radius 1 (structural diversity)",
    "FpDensityMorgan2":    "Morgan fingerprint density at radius 2",
    "FpDensityMorgan3":    "Morgan fingerprint density at radius 3",

    # MOE-style VSA descriptors (surface area within property ranges)
    "SlogP_VSA1":  "surface area of low-logP atoms (bin 1)",
    "SlogP_VSA2":  "surface area of low-logP atoms (bin 2)",
    "SlogP_VSA3":  "surface area of moderate-logP atoms (bin 3)",
    "SlogP_VSA4":  "surface area of moderate-logP atoms (bin 4)",
    "SlogP_VSA5":  "surface area of moderate-logP atoms (bin 5)",
    "SlogP_VSA6":  "surface area of high-logP atoms (bin 6)",
    "SlogP_VSA7":  "surface area of high-logP atoms (bin 7)",
    "SlogP_VSA8":  "surface area of very high-logP atoms (bin 8)",
    "SlogP_VSA9":  "surface area of very high-logP atoms (bin 9)",
    "SlogP_VSA10": "surface area of highest-logP atoms (bin 10)",
    "SlogP_VSA11": "surface area of highest-logP atoms (bin 11)",
    "SlogP_VSA12": "surface area of highest-logP atoms (bin 12)",
    "SMR_VSA1":  "surface area of low molar refractivity atoms (bin 1)",
    "SMR_VSA2":  "surface area of low MR atoms (bin 2)",
    "SMR_VSA3":  "surface area of moderate MR atoms (bin 3)",
    "SMR_VSA4":  "surface area of moderate MR atoms (bin 4)",
    "SMR_VSA5":  "surface area of moderate-high MR atoms (bin 5)",
    "SMR_VSA6":  "surface area of high MR atoms (bin 6)",
    "SMR_VSA7":  "surface area of high MR atoms (bin 7)",
    "SMR_VSA8":  "surface area of very high MR atoms (bin 8)",
    "SMR_VSA9":  "surface area of highest MR atoms (bin 9)",
    "SMR_VSA10": "surface area of highest MR atoms (bin 10)",
    "PEOE_VSA1":  "surface area of strongly negative partial charge atoms (bin 1)",
    "PEOE_VSA2":  "surface area of negative charge atoms (bin 2)",
    "PEOE_VSA3":  "surface area of slightly negative charge atoms (bin 3)",
    "PEOE_VSA4":  "surface area of slightly negative charge atoms (bin 4)",
    "PEOE_VSA5":  "surface area of near-neutral atoms (bin 5)",
    "PEOE_VSA6":  "surface area of near-neutral atoms (bin 6)",
    "PEOE_VSA7":  "surface area of slightly positive charge atoms (bin 7)",
    "PEOE_VSA8":  "surface area of slightly positive charge atoms (bin 8)",
    "PEOE_VSA9":  "surface area of positive charge atoms (bin 9)",
    "PEOE_VSA10": "surface area of positive charge atoms (bin 10)",
    "PEOE_VSA11": "surface area of strongly positive charge atoms (bin 11)",
    "PEOE_VSA12": "surface area of strongly positive charge atoms (bin 12)",
    "PEOE_VSA13": "surface area of very strongly positive atoms (bin 13)",
    "PEOE_VSA14": "surface area of very strongly positive atoms (bin 14)",
    "EState_VSA1":  "surface area in lowest E-state range (bin 1)",
    "EState_VSA2":  "surface area in low E-state range (bin 2)",
    "EState_VSA3":  "surface area in low-moderate E-state range (bin 3)",
    "EState_VSA4":  "surface area in moderate E-state range (bin 4)",
    "EState_VSA5":  "surface area in moderate E-state range (bin 5)",
    "EState_VSA6":  "surface area in moderate-high E-state range (bin 6)",
    "EState_VSA7":  "surface area in high E-state range (bin 7)",
    "EState_VSA8":  "surface area in high E-state range (bin 8)",
    "EState_VSA9":  "surface area in very high E-state range (bin 9)",
    "EState_VSA10": "surface area in highest E-state range (bin 10)",
    "EState_VSA11": "surface area in highest E-state range (bin 11)",
    "VSA_EState1":  "E-state value summed over low surface-area atoms (bin 1)",
    "VSA_EState2":  "E-state value summed over moderate surface-area atoms (bin 2)",
    "VSA_EState3":  "E-state value summed over moderate surface-area atoms (bin 3)",
    "VSA_EState4":  "E-state value summed over moderate surface-area atoms (bin 4)",
    "VSA_EState5":  "E-state value summed over moderate surface-area atoms (bin 5)",
    "VSA_EState6":  "E-state value summed over moderate surface-area atoms (bin 6)",
    "VSA_EState7":  "E-state value summed over high surface-area atoms (bin 7)",
    "VSA_EState8":  "E-state value summed over high surface-area atoms (bin 8)",
    "VSA_EState9":  "E-state value summed over high surface-area atoms (bin 9)",
    "VSA_EState10": "E-state value summed over highest surface-area atoms (bin 10)",

    # BCUT2D
    "BCUT2D_MWHI":     "BCUT2D: highest eigenvalue weighted by atomic mass",
    "BCUT2D_MWLOW":    "BCUT2D: lowest eigenvalue weighted by atomic mass",
    "BCUT2D_CHGHI":    "BCUT2D: highest eigenvalue weighted by partial charge",
    "BCUT2D_CHGLO":    "BCUT2D: lowest eigenvalue weighted by partial charge",
    "BCUT2D_LOGPHI":   "BCUT2D: highest eigenvalue weighted by atomic logP",
    "BCUT2D_LOGPLOW":  "BCUT2D: lowest eigenvalue weighted by atomic logP",
    "BCUT2D_MRHI":     "BCUT2D: highest eigenvalue weighted by molar refractivity",
    "BCUT2D_MRLOW":    "BCUT2D: lowest eigenvalue weighted by molar refractivity",

    # Common fragment counts
    "fr_NH0":              "count of tertiary amine groups (no N-H)",
    "fr_NH1":              "count of secondary amine groups (one N-H)",
    "fr_NH2":              "count of primary amine groups (two N-H)",
    "fr_quatN":            "count of quaternary (positively charged) nitrogen groups",
    "fr_amide":            "count of amide groups",
    "fr_amidine":          "count of amidine groups",
    "fr_aniline":          "count of aniline groups (aromatic amine on benzene)",
    "fr_pyridine":         "count of pyridine groups",
    "fr_imidazole":        "count of imidazole groups",
    "fr_oxazole":          "count of oxazole groups",
    "fr_thiazole":         "count of thiazole groups",
    "fr_furan":            "count of furan groups",
    "fr_thiophene":        "count of thiophene groups",
    "fr_pyrrole":          "count of pyrrole groups",
    "fr_morpholine":       "count of morpholine groups",
    "fr_piperdine":        "count of piperidine groups",
    "fr_piperzine":        "count of piperazine groups",
    "fr_ester":            "count of ester groups",
    "fr_ether":            "count of ether groups",
    "fr_aldehyde":         "count of aldehyde groups",
    "fr_ketone":           "count of ketone groups",
    "fr_C_O":              "count of C=O groups (all kinds)",
    "fr_C_O_noCOO":        "count of C=O groups excluding carboxylic acids",
    "fr_COO":              "count of carboxylic acid groups",
    "fr_COO2":             "count of carboxylate anion groups",
    "fr_C_S":              "count of C=S groups",
    "fr_phenol":           "count of phenol groups (Ar-OH)",
    "fr_phenol_noOrthoHbond": "count of phenols without ortho H-bond donors",
    "fr_aryl_methyl":      "count of aryl methyl groups",
    "fr_alkyl_carbamate":  "count of alkyl carbamate groups",
    "fr_alkyl_halide":     "count of alkyl halide groups",
    "fr_aryl_halide":      "count of aryl halide groups",
    "fr_halogen":          "count of halogen atoms (F, Cl, Br, I)",
    "fr_nitro":            "count of nitro groups",
    "fr_nitro_arom":       "count of aromatic nitro groups",
    "fr_nitroso":          "count of nitroso groups",
    "fr_nitrile":          "count of nitrile (cyano) groups",
    "fr_sulfide":          "count of sulfide groups",
    "fr_sulfone":          "count of sulfone groups",
    "fr_sulfonamd":        "count of sulfonamide groups",
    "fr_benzene":          "count of benzene rings",
    "fr_Ar_N":             "count of aromatic nitrogens",
    "fr_Ar_NH":            "count of aromatic NH groups",
    "fr_Ar_OH":            "count of aromatic hydroxyls",
    "fr_Ar_COO":           "count of aromatic carboxylic acids",
    "fr_Al_OH":            "count of aliphatic hydroxyls",
    "fr_Al_OH_noTert":     "count of aliphatic hydroxyls excluding tertiary",
    "fr_Al_COO":           "count of aliphatic carboxylic acids",
    "fr_Ndealkylation1":   "count of N-dealkylation sites (type 1)",
    "fr_Ndealkylation2":   "count of N-dealkylation sites (type 2)",
    "fr_para_hydroxylation": "count of para hydroxylation sites",
    "fr_unbrch_alkane":    "count of unbranched alkane chains",
    "fr_methoxy":          "count of methoxy (-OCH3) groups",
    "fr_thiocyan":         "count of thiocyanate groups",
    "fr_isocyan":          "count of isocyanate groups",
    "fr_imide":            "count of imide groups",
    "fr_hdrzine":          "count of hydrazine groups",
    "fr_hdrzone":          "count of hydrazone groups",
    "fr_diazo":            "count of diazo groups",
    "fr_azo":              "count of azo groups (-N=N-)",
    "fr_azide":            "count of azide groups",
    "fr_lactam":           "count of lactam groups (cyclic amide)",
    "fr_lactone":          "count of lactone groups (cyclic ester)",
    "fr_epoxide":          "count of epoxide groups",
    "fr_oxime":            "count of oxime groups",
    "fr_priamide":         "count of primary amide groups",
    "fr_prisulfonamd":     "count of primary sulfonamide groups",
    "fr_HOCCN":            "count of HO-C-C-N motifs",
    "fr_Imine":            "count of imine groups (C=N)",
    "fr_ketone_Topliss":   "count of Topliss-style ketone groups",
    "fr_dihydropyridine":  "count of dihydropyridine groups",
    "fr_guanido":          "count of guanidine groups",
    "fr_phos_acid":        "count of phosphoric acid groups",
    "fr_phos_ester":       "count of phosphoric ester groups",
    "fr_barbitur":         "count of barbiturate groups",
    "fr_urea":             "count of urea groups",
    "fr_thiophene":        "count of thiophene groups",
    "fr_term_acetylene":   "count of terminal acetylene groups",
    "fr_tetrazole":        "count of tetrazole groups",
    "fr_triazole":         "count of triazole groups",
    "fr_oxazole":          "count of oxazole groups",
    "fr_isothiocyan":      "count of isothiocyanate groups",
    "fr_SH":               "count of thiol (S-H) groups",
    "fr_N_O":              "count of N-O bond groups",
    "fr_ArN":              "count of aromatic ring nitrogens",
}

# Default descriptor set used when no task-specific cache entry is found.
DEFAULT_RDKIT_FEATURES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "HeavyAtomCount", "qed",
]


_rdkit_feature_cache: Optional[Dict[str, Any]] = None
_descriptor_fn_cache: Optional[Dict[str, Any]] = None


def load_rdkit_feature_cache(path: Path = RDKIT_FEATURE_CACHE_PATH
                             ) -> Dict[str, Any]:
    """Load the per-task top-feature cache produced by
    compute_feature_correlation.py or compute_feature_importance.py."""
    global _rdkit_feature_cache
    if _rdkit_feature_cache is not None:
        return _rdkit_feature_cache
    if path.exists():
        with open(path) as f:
            _rdkit_feature_cache = json.load(f)
    else:
        _rdkit_feature_cache = {}
    return _rdkit_feature_cache


def _rdkit_descriptor_lookup():
    """Return {descriptor_name: callable_taking_mol} for all RDKit descriptors
    plus a few hand-added commonly-cited ones."""
    if not _RDKIT_AVAILABLE:
        return {}
    lookup = {name: fn for name, fn in Descriptors.descList}
    lookup.setdefault("qed", QED.qed)
    lookup.setdefault("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings)
    lookup.setdefault("NumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings)
    lookup.setdefault("HeavyAtomCount", lambda m: m.GetNumHeavyAtoms())
    return lookup


def _get_descriptor_fns():
    global _descriptor_fn_cache
    if _descriptor_fn_cache is None:
        _descriptor_fn_cache = _rdkit_descriptor_lookup()
    return _descriptor_fn_cache


def _format_descriptor(name: str, value: Any) -> str:
    """Format one descriptor as:
        '  name: value (qualitative_label) — explanation'
    Where:
      - qualitative_label appears for descriptors in _DESCRIPTOR_LABELS
      - explanation appears for descriptors in _DESCRIPTOR_DESCRIPTIONS
    Both are optional; if neither is present, just 'name: value'.
    """
    if value is None:
        return f"  {name}: n/a"
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return f"  {name}: {value}"
    # Integer formatting for counts
    if abs(fval - round(fval)) < 1e-6 and abs(fval) < 1e6:
        val_str = f"{int(round(fval))}"
    else:
        val_str = f"{fval:.3f}"

    parts = [f"  {name}: {val_str}"]
    if name in _DESCRIPTOR_LABELS:
        lo, hi, labels = _DESCRIPTOR_LABELS[name]
        parts.append(f"({_bucket(fval, lo, hi, labels)})")
    if name in _DESCRIPTOR_DESCRIPTIONS:
        parts.append(f"— {_DESCRIPTOR_DESCRIPTIONS[name]}")
    return " ".join(parts)


def rdkit_descriptors_text(smiles: str, dataset_name: str = "",
                           task_name: str = "",
                           top_k: Optional[int] = None) -> str:
    """Compute task-relevant RDKit descriptors and return them as a text block.

    Looks up cached top features for (dataset, task_name). For multi-task
    datasets (tox21/sider/toxcast), the cache key is 'dataset::sub_task_name'.
    Falls back to a hand-curated default set if not in cache.

    Args:
      smiles: input molecule SMILES
      dataset_name: 'bbbp', 'bace', etc. (used to look up task-specific features)
      task_name: sub-task name for multi-task datasets, ignored otherwise
      top_k: cap on number of features (None = use full cached list)

    Returns:
      Multi-line text block ready to inject into a prompt. Empty string on failure.
    """
    if not _RDKIT_AVAILABLE:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return ""
    if mol is None:
        return ""

    # Decide which descriptors to compute
    cache = load_rdkit_feature_cache()
    cache_key = f"{dataset_name}::{task_name}" if task_name else dataset_name
    entry = cache.get(cache_key) or cache.get(dataset_name)
    if entry and entry.get("top_features"):
        feature_names = [f["name"] for f in entry["top_features"]]
        if top_k is not None:
            feature_names = feature_names[:top_k]
        method = entry.get("method", "ranked")
        header = (f"Task-relevant molecular descriptors "
                  f"(top {len(feature_names)} by {method}):")
    else:
        feature_names = DEFAULT_RDKIT_FEATURES
        if top_k is not None:
            feature_names = feature_names[:top_k]
        header = "Molecular descriptors (computed via RDKit):"

    fn_lookup = _get_descriptor_fns()
    lines = [header]
    for name in feature_names:
        fn = fn_lookup.get(name)
        if fn is None:
            continue
        try:
            val = fn(mol)
        except Exception:
            continue
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            continue
        lines.append(_format_descriptor(name, val))

    # Always tack on Lipinski compliance (universally interpretable)
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lipinski = ("passes" if ro5_violations == 0
                    else f"{ro5_violations} violation(s)")
        lines.append(f"  Lipinski's Rule of 5: {lipinski}")
    except Exception:
        pass

    return "\n".join(lines) if len(lines) > 1 else ""


# ===========================================================================
# Combined block builder
# ===========================================================================
def build_prompt_injection(smiles: str, dataset_name: str, task_name: str,
                           syn_cache: Optional[Dict[str, Any]],
                           use_synonyms: bool, use_fgs: bool, use_rdkit: bool,
                           rdkit_top_k: Optional[int] = None
                           ) -> Dict[str, str]:
    """Build the three injection blocks based on the flag combination.

    Returns dict with keys 'synonyms_block', 'fgs_block', 'rdkit_block'.
    Empty strings for flags that are off OR if the relevant data is missing.
    """
    out = {"synonyms_block": "", "fgs_block": "", "rdkit_block": ""}
    if use_synonyms:
        out["synonyms_block"] = synonyms_text(smiles, syn_cache)
    if use_fgs:
        out["fgs_block"] = functional_groups_text(smiles)
    if use_rdkit:
        out["rdkit_block"] = rdkit_descriptors_text(
            smiles, dataset_name=dataset_name, task_name=task_name,
            top_k=rdkit_top_k)
    return out


def availability_check(use_synonyms: bool, use_fgs: bool, use_rdkit: bool,
                       syn_cache_path: Path = DEFAULT_SYNONYM_CACHE) -> None:
    """Raise informative errors if requested capabilities are missing."""
    if use_synonyms and not syn_cache_path.exists():
        raise FileNotFoundError(
            f"--prompt_synonyms requires LLM-filtered cache at {syn_cache_path}\n"
            f"Run build_synonym_cache.py first.")
    if use_fgs and not _ACCFG_AVAILABLE:
        raise RuntimeError(
            "--prompt_fgs requires AccFG. Install with `pip install accfg`.")
    if use_rdkit and not _RDKIT_AVAILABLE:
        raise RuntimeError(
            "--prompt_rdkit requires RDKit. Install with `pip install rdkit`.")
    if use_rdkit and not RDKIT_FEATURE_CACHE_PATH.exists():
        print(f"NOTE: no RDKit feature cache at {RDKIT_FEATURE_CACHE_PATH}.")
        print(f"      --prompt_rdkit will use a default hand-curated set.")
        print(f"      Run compute_feature_correlation.py to build per-task ranking.")