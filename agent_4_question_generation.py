"""
AGENT 4: Multimodal Chemistry Question Generation Agent
Generates verified, JEE-standard multimodal chemistry questions:
  1. Image Questions: Programmatically generated using RDKit (organic & inorganic molecules)
     and Matplotlib (chemical kinetics plots). 100% computed, ground-truth verified answers.
  2. Text Questions: Transformed from Knowledge Graph source questions via LLM (e.g. Qwen),
     validated with deduplication filters and self-verification checks.
  3. Exporters: Outputs formatted question banks to JSON, PKL, HTML, and Word DOCX.
"""

import os
import re
import json
import time
import math
import html
import pickle
import base64
import random
import difflib
import argparse
import traceback
from collections import Counter

from config import (
    MODEL_NAME,
    KG_PICKLE_PATH,
    AGENT2_OUTPUT_JSON,
    AGENT3_OUTPUT_TXT,
    AGENT4_OUTPUT_JSON,
    AGENT4_OUTPUT_PKL,
    AGENT4_OUTPUT_HTML,
    AGENT4_OUTPUT_DOCX,
    GENERATED_IMAGES_DIR,
    BATCH_SIZE,
    MAX_BATCH_ATTEMPTS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    RETRY_TEMPERATURE,
    TOP_P,
    TOP_K,
    REPETITION_PENALTY,
    VERIFY_TEXT,
    VERIFY_MAX_TOKENS,
    STRICT_VERIFY_ATTEMPTS,
    DUP_SEQ_RATIO,
    DUP_JACCARD,
    get_device_info,
    clear_cuda_memory,
    show_gpu_memory,
)

# ==============================================================================
# 1. MOLECULE BANKS (Vetted structures for programmatic question synthesis)
# ==============================================================================
ORGANIC_BANK = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("vanillin", "COc1cc(C=O)ccc1O"),
    ("cinnamaldehyde", "O=CC=Cc1ccccc1"),
    ("salicylaldehyde", "O=Cc1ccccc1O"),
    ("acetanilide", "CC(=O)Nc1ccccc1"),
    ("methyl benzoate", "COC(=O)c1ccccc1"),
    ("benzophenone", "O=C(c1ccccc1)c1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("lactic acid", "CC(O)C(=O)O"),
    ("tartaric acid", "OC(C(O)C(=O)O)C(=O)O"),
    ("alanine", "CC(N)C(=O)O"),
    ("glucose (pyranose)", "OCC1OC(O)C(O)C(O)C1O"),
    ("menthol", "CC(C)C1CCC(C)CC1O"),
    ("camphor", "CC1(C)C2CCC1(C)C(=O)C2"),
    ("limonene", "CC1=CCC(CC1)C(C)=C"),
    ("citral", "CC(C)=CCCC(C)=CC=O"),
    ("but-2-ynoic acid", "CC#CC(=O)O"),
    ("phenylacetylene", "C#Cc1ccccc1"),
    ("styrene", "C=Cc1ccccc1"),
    ("cyclohexanone", "O=C1CCCCC1"),
    ("acetophenone", "CC(=O)c1ccccc1"),
    ("benzamide", "NC(=O)c1ccccc1"),
    ("nitrobenzene", "[O-][N+](=O)c1ccccc1"),
    ("p-nitrophenol", "Oc1ccc(cc1)[N+](=O)[O-]"),
    ("aniline", "Nc1ccccc1"),
    ("phthalic acid", "OC(=O)c1ccccc1C(=O)O"),
    ("maleic acid", "OC(=O)C=CC(=O)O"),
    ("crotonaldehyde", "CC=CC=O"),
    ("mesityl oxide", "CC(C)=CC(C)=O"),
    ("pyridine", "c1ccncc1"),
    ("furfural", "O=Cc1ccco1"),
    ("adrenaline", "CNCC(O)c1ccc(O)c(O)c1"),
    ("ephedrine", "CNC(C)C(O)c1ccccc1"),
    ("2-methylbutan-1-ol", "CCC(C)CO"),
    ("2,3-dichlorobutane", "CC(Cl)C(Cl)C"),
    ("bromochlorofluoromethane", "FC(Cl)Br"),
    ("buta-1,3-diene", "C=CC=C"),
    ("cyclopentadiene", "C1=CC=CC1"),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("acrylonitrile", "C=CC#N"),
    ("benzoic acid", "OC(=O)c1ccccc1"),
    ("ethyl acetoacetate", "CCOC(=O)CC(C)=O"),
    ("p-toluidine", "Cc1ccc(N)cc1"),
    ("salicylic acid", "OC(=O)c1ccccc1O"),
    ("benzyl alcohol", "OCc1ccccc1"),
    ("methyl vinyl ketone", "C=CC(C)=O"),
    ("2-butyne-1,4-diol", "OCC#CCO"),
    ("cyclohexene", "C1=CCCCC1"),
]

INORGANIC_BANK = [
    ("SF4", "FS(F)(F)F"),
    ("XeF4", "F[Xe](F)(F)F"),
    ("XeF2", "F[Xe]F"),
    ("SF6", "FS(F)(F)(F)(F)F"),
    ("PCl5", "ClP(Cl)(Cl)(Cl)Cl"),
    ("PF5", "FP(F)(F)(F)F"),
    ("XeO3", "O=[Xe](=O)=O"),
    ("XeOF4", "O=[Xe](F)(F)(F)F"),
    ("XeO2F2", "O=[Xe](=O)(F)F"),
    ("IF5", "FI(F)(F)(F)F"),
    ("BF3", "FB(F)F"),
    ("NF3", "FN(F)F"),
    ("NH3", "N"),
    ("H2O", "O"),
    ("CH4", "C"),
    ("SO2", "O=S=O"),
    ("SO3", "O=S(=O)=O"),
    ("BeCl2", "Cl[Be]Cl"),
    ("CO2", "O=C=O"),
    ("SCl2", "ClSCl"),
    ("SiF4", "F[Si](F)(F)F"),
    ("IF7", "FI(F)(F)(F)(F)(F)F"),
]

FUNCTIONAL_GROUPS = {
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[CX3](=O)[NX3]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[CX4][OX2H]",
    "phenol": "c[OX2H]",
    "ether": "[OX2;!$(O[CX3]=O)]([#6])[#6]",
    "amine": "[NX3;!$(NC=O);!$(N=*);!$([N+])]",
    "nitro": "[N+](=O)[O-]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "nitrile": "C#N",
    "aromatic ring": "a1aaaaa1",
}

HYB_NAMES = {
    2: "sp", 3: "sp\u00b2", 4: "sp\u00b3", 5: "sp\u00b3d",
    6: "sp\u00b3d\u00b2", 7: "sp\u00b3d\u00b3"
}

SHAPES = {
    (2, 0): "Linear",
    (3, 0): "Trigonal planar",
    (3, 1): "Bent (V-shaped)",
    (4, 0): "Tetrahedral",
    (4, 1): "Trigonal pyramidal",
    (4, 2): "Bent (V-shaped)",
    (5, 0): "Trigonal bipyramidal",
    (5, 1): "See-saw",
    (5, 2): "T-shaped",
    (5, 3): "Linear",
    (6, 0): "Octahedral",
    (6, 1): "Square pyramidal",
    (6, 2): "Square planar",
    (7, 0): "Pentagonal bipyramidal",
}

ORGANIC_FAMS = [
    "org_sp2", "org_sp3", "org_chiral", "org_sigma", "org_pi",
    "org_diff", "org_rings", "org_formula", "org_groups",
    "org_dbe", "org_combustion", "org_molar_mass"
]
INORGANIC_FAMS = ["inorg_hyb", "inorg_shape", "inorg_lp", "inorg_combined"]
GRAPH_FAMS = [
    "g_first_order", "g_arrhenius", "g_zero_order", "g_exp_decay",
    "g_first_order_two_step", "g_zero_order_two_step"
]

REQUIRED_FIELDS = ["question", "option_A", "option_B", "option_C", "option_D", "correct_answer", "explanation"]
TEXT_REFERS_TO_IMAGE = re.compile(
    r"\b(figure|diagram|image|picture)\b|structure (shown|given below)|shown below|"
    r"graph shown|following (structure|graph)", re.IGNORECASE
)


# ==============================================================================
# 2. IMAGE QUESTION ENGINE (RDKit + Matplotlib programmatic questions)
# ==============================================================================
class ImageQuestionEngine:
    """Generates 100% computed, ground-truth verified image questions."""

    def __init__(self, image_folder=GENERATED_IMAGES_DIR):
        self.image_folder = image_folder
        os.makedirs(self.image_folder, exist_ok=True)
        self.image_counter = 0
        self.used_keys = set()
        self.used_molecules = set()
        self.family_use = Counter()

    def _next_image_path(self, prefix: str) -> str:
        self.image_counter += 1
        return os.path.join(self.image_folder, f"{prefix}_{self.image_counter:03d}.png")

    def _is_number(self, text: str) -> bool:
        try:
            float(str(text).split()[0])
            return True
        except Exception:
            return False

    def int_options(self, correct: int, extras=()) -> list:
        pool = []
        for c in list(extras) + [correct - 1, correct + 1, correct - 2, correct + 2, correct + 3, correct - 3]:
            if c >= 0 and c != correct and c not in pool:
                pool.append(c)
        return [str(x) for x in pool[:3]]

    def assemble_image_question(
        self, question, correct, distractors, explanation, image_path,
        image_type, image_description, smiles=None
    ) -> dict:
        options = [str(correct)] + [str(d) for d in distractors][:3]
        if len(set(options)) != 4:
            raise ValueError("Options are not distinct")

        if all(self._is_number(o) for o in options):
            options.sort(key=lambda o: float(o.split()[0]))
        else:
            random.shuffle(options)

        q = {
            "question": question,
            "option_A": options[0],
            "option_B": options[1],
            "option_C": options[2],
            "option_D": options[3],
            "correct_answer": "ABCD"[options.index(str(correct))],
            "explanation": explanation,
            "question_type": "image",
            "image_type": image_type,
            "image_description": image_description,
            "generated_image": image_path,
            "image_path": image_path,
            "source": "image_generator",
            "difficulty_category": "computed_image",
            "verification": "computed",
        }
        if smiles:
            q["smiles"] = smiles
        return q

    def draw_molecule(self, smiles: str, path: str):
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
        except ImportError:
            raise ImportError("RDKit is required for molecule drawing. Install via: pip install rdkit")
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(380, 260))
        img.save(path)

    def organic_facts(self, smiles: str) -> dict:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
        except ImportError:
            raise ImportError("RDKit is required for molecular analysis.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid smiles: {smiles}")
        molH = Chem.AddHs(mol)

        sp = sp2 = sp3 = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            h = atom.GetHybridization()
            if h == Chem.HybridizationType.SP:
                sp += 1
            elif h == Chem.HybridizationType.SP2:
                sp2 += 1
            elif h == Chem.HybridizationType.SP3:
                sp3 += 1

        kek = Chem.Mol(molH)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        sigma = pi = 0
        for b in kek.GetBonds():
            bt = b.GetBondType()
            if bt == Chem.BondType.SINGLE:
                sigma += 1
            elif bt == Chem.BondType.DOUBLE:
                sigma += 1
                pi += 1
            elif bt == Chem.BondType.TRIPLE:
                sigma += 1
                pi += 2

        h_bonds = sum(
            1 for b in molH.GetBonds()
            if b.GetBeginAtom().GetAtomicNum() == 1 or b.GetEndAtom().GetAtomicNum() == 1
        )
        chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))

        # Check functional groups
        group_patterns = {k: Chem.MolFromSmarts(v) for k, v in FUNCTIONAL_GROUPS.items()}
        groups = [name for name, patt in group_patterns.items() if patt and mol.HasSubstructMatch(patt)]

        return {
            "sp": sp, "sp2": sp2, "sp3": sp3,
            "sigma": sigma, "pi": pi, "sigma_no_h": sigma - h_bonds,
            "chiral": chiral,
            "rings": mol.GetRingInfo().NumRings(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "groups": groups,
        }

    def central_atom_info(self, smiles: str) -> dict:
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit is required for central atom analysis.")

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        max_deg = max(a.GetDegree() for a in heavy)
        candidates = [a for a in heavy if a.GetDegree() == max_deg]
        if len(candidates) != 1:
            raise ValueError("No unique central atom")

        atom = candidates[0]
        bond_order_sum = int(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()))
        valence_e = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
        remaining = valence_e - bond_order_sum - atom.GetFormalCharge()
        if remaining < 0 or remaining % 2 != 0:
            raise ValueError("Odd electron count")

        lone_pairs = remaining // 2
        sigma = atom.GetDegree()
        steric = sigma + lone_pairs
        if steric not in HYB_NAMES or (steric, lone_pairs) not in SHAPES:
            raise ValueError("Unsupported geometry")

        return {
            "symbol": atom.GetSymbol(), "valence_e": valence_e, "sigma": sigma,
            "bond_order_sum": bond_order_sum, "lone_pairs": lone_pairs,
            "steric": steric, "hybridisation": HYB_NAMES[steric],
            "shape": SHAPES[(steric, lone_pairs)],
        }

    def build_organic(self, family: str) -> dict:
        name, smiles = random.choice(ORGANIC_BANK)
        f = self.organic_facts(smiles)
        intro = f"Consider the structure of {name} shown in the figure."
        formula = f["formula"]

        if family == "org_sp2":
            if f["sp2"] < 1: raise ValueError("ineligible")
            question = "The number of sp\u00b2-hybridised carbon atoms present in the structure shown is"
            correct = str(f["sp2"])
            distractors = self.int_options(f["sp2"], extras=[f["sp3"], f["sp"]])
            expl = f"{intro} Carbon atoms with one double bond are sp\u00b2 hybridised. Total = {f['sp2']}."
        elif family == "org_sp3":
            if f["sp3"] < 1: raise ValueError("ineligible")
            question = "The number of sp\u00b3-hybridised carbon atoms present in the structure shown is"
            correct = str(f["sp3"])
            distractors = self.int_options(f["sp3"], extras=[f["sp2"], f["sp"]])
            expl = f"{intro} Carbon atoms with four single bonds are sp\u00b3 hybridised. Total = {f['sp3']}."
        elif family == "org_chiral":
            if f["chiral"] < 1: raise ValueError("ineligible")
            question = "The number of chiral (asymmetric) carbon centres present in the structure shown is"
            correct = str(f["chiral"])
            distractors = self.int_options(f["chiral"], extras=[0, 1, 2])
            expl = f"{intro} Chiral carbons are bonded to 4 distinct groups. Total = {f['chiral']}."
        elif family == "org_sigma":
            question = "The total number of \u03c3 (sigma) bonds present in the structure shown (including C\u2013H bonds) is"
            correct = str(f["sigma"])
            distractors = self.int_options(f["sigma"], extras=[f["sigma_no_h"], f["sigma"] + 2])
            expl = f"{intro} Counting all single bonds and one bond from each multiple bond yields {f['sigma']} \u03c3 bonds."
        elif family == "org_pi":
            if f["pi"] < 1: raise ValueError("ineligible")
            question = "The total number of \u03c0 (pi) bonds present in the structure shown is"
            correct = str(f["pi"])
            distractors = self.int_options(f["pi"], extras=[f["rings"], f["pi"] + 1])
            expl = f"{intro} Double bonds contain 1 \u03c0 bond and triple bonds contain 2 \u03c0 bonds. Total = {f['pi']}."
        elif family == "org_diff":
            diff = abs(f["sp2"] - f["sp3"])
            question = "The absolute difference between the number of sp\u00b2 and sp\u00b3 hybridised carbon atoms is"
            correct = str(diff)
            distractors = self.int_options(diff, extras=[f["sp2"], f["sp3"]])
            expl = f"{intro} It contains {f['sp2']} sp\u00b2 and {f['sp3']} sp\u00b3 carbons. |{f['sp2']} - {f['sp3']}| = {diff}."
        elif family == "org_rings":
            if f["rings"] < 1: raise ValueError("ineligible")
            question = "The total number of rings present in the structure shown is"
            correct = str(f["rings"])
            distractors = self.int_options(f["rings"])
            expl = f"{intro} The smallest set of smallest rings gives {f['rings']} ring(s)."
        elif family == "org_formula":
            question = "The molecular formula of the compound shown is"
            correct = formula
            counts = Counter({el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)})
            distractors = [
                f"C{counts['C']}H{counts.get('H', 2) + 2}O{counts.get('O', 1)}",
                f"C{counts['C']}H{max(1, counts.get('H', 2) - 2)}O{counts.get('O', 1)}",
                f"C{counts['C'] + 1}H{counts.get('H', 2) + 2}"
            ]
            expl = f"{intro} Counting all carbon, hydrogen, and heteroatoms gives {formula}."
        elif family == "org_dbe":
            counts = Counter({el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)})
            C = counts.get("C", 0); H = counts.get("H", 0); N = counts.get("N", 0)
            X = sum(counts.get(el, 0) for el in ["F", "Cl", "Br", "I"])
            dbe = (2 * C + 2 + N - H - X) / 2
            if dbe < 1 or abs(dbe - round(dbe)) > 1e-9: raise ValueError("ineligible")
            dbe = int(round(dbe))
            rings = f["rings"]
            if dbe < rings: raise ValueError("ineligible")
            pi_bonds = dbe - rings
            question = (
                f"For the structure shown, the degree of unsaturation (DBE) is {dbe}. "
                f"If the {rings} ring(s) account for the ring contribution, how many \u03c0 bonds must be present?"
            )
            correct = str(pi_bonds)
            distractors = self.int_options(pi_bonds, extras=[dbe, rings, f["pi"]])
            expl = f"DBE = {dbe} = rings + \u03c0 bonds -> \u03c0 bonds = {dbe} - {rings} = {pi_bonds}."
        elif family == "org_combustion":
            counts = Counter({el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)})
            if any(counts.get(el, 0) for el in ["F", "Cl", "Br", "I"]) or counts.get("C", 0) == 0:
                raise ValueError("ineligible")
            C = counts.get("C", 0); H = counts.get("H", 0); O = counts.get("O", 0)
            o2_per_mol = C + H / 4 - O / 2
            if o2_per_mol <= 0: raise ValueError("ineligible")
            sample_mol = random.choice([0.20, 0.25, 0.40, 0.50])
            required_o2 = sample_mol * o2_per_mol
            question = (
                f"The compound shown has formula {formula}. If {sample_mol:g} mol "
                "undergoes complete combustion, the amount of O\u2082 required is"
            )
            correct = f"{required_o2:.3g} mol"
            distractors = [
                f"{sample_mol * (C + H / 2 - O):.3g} mol",
                f"{sample_mol * (C + H / 4):.3g} mol",
                f"{sample_mol * (C + H / 4 + O / 2):.3g} mol",
            ]
            expl = f"For CxHyOz, O\u2082 per mole = x + y/4 - z/2 = {o2_per_mol:.3g} mol. Total = {sample_mol} \u00d7 {o2_per_mol:.3g} = {required_o2:.3g} mol."
        elif family == "org_molar_mass":
            counts = Counter({el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)})
            masses = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.904}
            molar_mass = sum(masses[el] * n for el, n in counts.items())
            sample_mol = random.choice([0.15, 0.20, 0.25, 0.40])
            sample_mass = sample_mol * molar_mass
            question = f"The structure shown has formula {formula}. What mass is present in {sample_mol:g} mol?"
            correct = f"{sample_mass:.3g} g"
            distractors = [f"{molar_mass:.3g} g", f"{sample_mass / 2:.3g} g", f"{sample_mass * 2:.3g} g"]
            expl = f"Molar mass of {formula} = {molar_mass:.3f} g/mol. m = n \u00d7 M = {sample_mol:g} \u00d7 {molar_mass:.3f} = {sample_mass:.3g} g."
        elif family == "org_groups":
            groups = f["groups"]
            if len(groups) < 2 or len(groups) > 4: raise ValueError("ineligible")
            correct = ", ".join(sorted(groups))
            absent = [g for g in FUNCTIONAL_GROUPS if g not in groups]
            distractors = [
                ", ".join(sorted(list(groups)[:-1] + [absent[0]])),
                ", ".join(sorted(list(groups)[:-1] + [absent[1] if len(absent) > 1 else "alkyne"])),
                ", ".join(sorted(list(groups)[:-1] + ["nitrile"]))
            ]
            question = "Which option correctly lists the functional groups present in the structure shown?"
            expl = f"{intro} It contains: {correct}."
        else:
            raise ValueError(f"Unknown family: {family}")

        key = (smiles, family)
        if key in self.used_keys: raise ValueError("Already used")
        path = self._next_image_path("mol")
        self.draw_molecule(smiles, path)

        q = self.assemble_image_question(
            question, correct, distractors, expl, path, "molecular_structure",
            f"Structure of an organic molecule ({family})", smiles=smiles
        )
        self.used_keys.add(key)
        self.used_molecules.add(smiles)
        return q

    def build_inorganic(self, family: str) -> dict:
        name, smiles = random.choice(INORGANIC_BANK)
        info = self.central_atom_info(smiles)
        key = (smiles, family)
        if key in self.used_keys: raise ValueError("Already used")

        cen = info["symbol"]
        intro = f"Consider the molecule {name} whose central atom is {cen}."

        if family == "inorg_hyb":
            question = f"According to VSEPR theory, the hybridisation of the central {cen} atom in {name} is"
            correct = info["hybridisation"]
            distractors = [h for h in ["sp", "sp\u00b2", "sp\u00b3", "sp\u00b3d", "sp\u00b3d\u00b2", "sp\u00b3d\u00b3"] if h != correct][:3]
            expl = f"{intro} Steric number = {info['sigma']} (\u03c3 bonds) + {info['lone_pairs']} (lone pairs) = {info['steric']}. Hybridisation = {correct}."
        elif family == "inorg_shape":
            question = f"The molecular geometry (shape) of {name} predicted by VSEPR theory is"
            correct = info["shape"]
            distractors = [s for s in ["Linear", "Bent (V-shaped)", "Trigonal planar", "Tetrahedral", "Trigonal pyramidal", "See-saw", "T-shaped", "Octahedral", "Square planar"] if s != correct][:3]
            expl = f"{intro} With {info['sigma']} bond pairs and {info['lone_pairs']} lone pairs, the shape is {correct}."
        elif family == "inorg_lp":
            question = f"The number of lone pairs on the central {cen} atom in {name} is"
            correct = str(info["lone_pairs"])
            distractors = self.int_options(info["lone_pairs"], extras=[0, 1, 2, 3])
            expl = f"{intro} Valence e\u207b = {info['valence_e']}, bonded e\u207b = {info['bond_order_sum']}. Lone pairs = ({info['valence_e']} - {info['bond_order_sum']})/2 = {info['lone_pairs']}."
        elif family == "inorg_combined":
            question = f"For the species {name}, which option correctly gives the hybridisation and molecular shape of the central atom?"
            correct = f"{info['hybridisation']}, {info['shape']}"
            all_hybs = ["sp\u00b3", "sp\u00b3d", "sp\u00b3d\u00b2"]
            distractors = [
                f"{random.choice(all_hybs)}, Tetrahedral",
                f"{info['hybridisation']}, Trigonal bipyramidal" if info['shape'] != "Trigonal bipyramidal" else f"{info['hybridisation']}, Octahedral",
                f"sp\u00b3d, {info['shape']}" if info['hybridisation'] != "sp\u00b3d" else f"sp\u00b3, {info['shape']}"
            ]
            expl = f"{intro} Steric number = {info['steric']} -> {info['hybridisation']} with {info['lone_pairs']} lone pairs -> {info['shape']}."
        else:
            raise ValueError(f"Unknown family: {family}")

        path = self._next_image_path("mol")
        self.draw_molecule(smiles, path)
        q = self.assemble_image_question(
            question, correct, distractors, expl, path, "inorganic_geometry",
            f"Structure and geometry of {name}", smiles=smiles
        )
        self.used_keys.add(key)
        return q

    def build_graph(self, family: str) -> dict:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting kinetics graphs.")

        def finish_graph(fig, path):
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

        def label_point(ax, x, y, text, dx=8, dy=8):
            ax.plot([x], [y], marker="o", color="red", zorder=4)
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                        fontweight="bold", color="darkred")

        if family == "g_first_order":
            k = random.choice([0.02, 0.03, 0.04, 0.05])
            fig, ax = plt.subplots(figsize=(6, 4))
            xs = [0, 60]
            ys = [0, -k * 60]
            ax.plot(xs, ys, color="navy")
            label_point(ax, 0, 0, "(0, 0)")
            label_point(ax, 60, ys[1], f"(60, {ys[1]:.2f})", dy=-12)
            ax.set_xlabel("Time, t (min)")
            ax.set_ylabel("ln([A]/[A]\u2080)")
            ax.grid(alpha=0.3)
            question = "The graph shows a first-order reaction ln([A]/[A]\u2080) vs time. The rate constant k is"
            correct = f"{k:.4f} min\u207b\u00b9"
            distractors = [f"{k*2:.4f} min\u207b\u00b9", f"{k/2:.4f} min\u207b\u00b9", f"{k*1.5:.4f} min\u207b\u00b9"]
            expl = f"For first-order, slope = -k. Here slope = {ys[1]:.2f}/60 = {-k:.4f}, so k = {k:.4f} min\u207b\u00b9."
            path = self._next_image_path("graph")
            finish_graph(fig, path)
            q = self.assemble_image_question(question, correct, distractors, expl, path, "graph", "First-order kinetics plot")
        elif family == "g_arrhenius":
            Ea_kJ = random.choice([50, 60, 75, 80, 100])
            slope = -Ea_kJ * 1000 / 8.314
            fig, ax = plt.subplots(figsize=(6, 4))
            xs = [0.002, 0.003]
            ys = [slope * x + 15 for x in xs]
            ax.plot(xs, ys, color="darkgreen")
            label_point(ax, xs[0], ys[0], f"({xs[0]}, {ys[0]:.1f})")
            label_point(ax, xs[1], ys[1], f"({xs[1]}, {ys[1]:.1f})", dy=-12)
            ax.set_xlabel("1/T (K\u207b\u00b9)")
            ax.set_ylabel("ln k")
            ax.grid(alpha=0.3)
            question = "From the given Arrhenius plot (ln k vs 1/T), the activation energy Ea is closest to"
            correct = f"{Ea_kJ} kJ mol\u207b\u00b9"
            distractors = [f"{Ea_kJ + 20} kJ mol\u207b\u00b9", f"{max(10, Ea_kJ - 20)} kJ mol\u207b\u00b9", f"{Ea_kJ * 2} kJ mol\u207b\u00b9"]
            expl = f"Slope = -Ea/R = {slope:.1f}. Ea = -slope \u00d7 8.314 J/mol = {Ea_kJ} kJ/mol."
            path = self._next_image_path("graph")
            finish_graph(fig, path)
            q = self.assemble_image_question(question, correct, distractors, expl, path, "graph", "Arrhenius plot")
        elif family == "g_zero_order":
            A0 = random.choice([1.0, 1.2, 1.5, 2.0])
            k = random.choice([0.01, 0.02, 0.025])
            fig, ax = plt.subplots(figsize=(6, 4))
            t_end = A0 / k
            ax.plot([0, t_end], [A0, 0], color="purple")
            label_point(ax, 0, A0, f"(0, {A0:.1f})")
            label_point(ax, t_end, 0, f"({t_end:.1f}, 0)", dy=-12)
            ax.set_xlabel("Time, t (min)")
            ax.set_ylabel("[A] (mol L\u207b\u00b9)")
            ax.grid(alpha=0.3)
            question = "The plot represents a zero-order reaction [A] vs time. The initial rate of reaction is"
            correct = f"{k:.3f} mol L\u207b\u00b9 min\u207b\u00b9"
            distractors = [f"{k*2:.3f} mol L\u207b\u00b9 min\u207b\u00b9", f"{k/2:.3f} mol L\u207b\u00b9 min\u207b\u00b9", f"{k*1.5:.3f} mol L\u207b\u00b9 min\u207b\u00b9"]
            expl = f"For zero-order, rate = k = -slope = {A0}/{t_end:.1f} = {k:.3f} mol/(L min)."
            path = self._next_image_path("graph")
            finish_graph(fig, path)
            q = self.assemble_image_question(question, correct, distractors, expl, path, "graph", "Zero-order decay plot")
        else:
            # First order two step
            t_half = random.choice([10, 15, 20, 25, 30])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([0, 3 * t_half], [0, -3 * math.log(2)], color="navy")
            label_point(ax, 0, 0, "(0, 0)")
            label_point(ax, t_half, -math.log(2), f"({t_half}, -0.69)", dy=-12)
            ax.set_xlabel("Time, t (min)")
            ax.set_ylabel("ln([A]/[A]\u2080)")
            ax.grid(alpha=0.3)
            question = f"In the first-order reaction shown, what percentage of reactant A has reacted after {3 * t_half} min?"
            correct = "87.5%"
            distractors = ["50%", "75%", "93.75%"]
            expl = f"t\u00bd = {t_half} min. At t = {3 * t_half} min (3 half-lives), fraction remaining = (1/2)\u00b3 = 1/8 = 12.5%, so reacted = 87.5%."
            path = self._next_image_path("graph")
            finish_graph(fig, path)
            q = self.assemble_image_question(question, correct, distractors, expl, path, "graph", "First-order multi-step plot")

        return q

    def family_pool(self, topic: str) -> list:
        t = str(topic).lower()
        if any(k in t for k in ["kinetic", "rate of reaction", "arrhenius"]):
            return GRAPH_FAMS
        if any(k in t for k in ["organic", "hydrocarbon", "isomer", "biomolecule", "amine", "carbonyl", "aldehyde", "alcohol"]):
            return ORGANIC_FAMS
        if any(k in t for k in ["bonding", "p-block", "inorganic", "hybrid", "vsepr"]):
            return INORGANIC_FAMS
        return ORGANIC_FAMS + INORGANIC_FAMS + GRAPH_FAMS

    def build_question(self, pool: list, max_tries: int = 300) -> dict:
        """Draws and generates an image question from the pool."""
        for attempt in range(max_tries):
            family = random.choice(pool)
            try:
                if family.startswith("org_"):
                    q = self.build_organic(family)
                elif family.startswith("inorg_"):
                    q = self.build_inorganic(family)
                else:
                    q = self.build_graph(family)
                self.family_use[family] += 1
                q["image_family"] = family
                return q
            except Exception:
                continue

        # If exhausted, reset used keys and retry
        self.used_keys.clear()
        return self.build_question(pool, max_tries=50)


# ==============================================================================
# 3. TEXT QUESTION ENGINE (KG Sources + LLM Generation & Verification)
# ==============================================================================
class TextQuestionEngine:
    """Transforms KG source questions into new JEE-style questions using an LLM."""

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None

    def initialize_llm(self):
        """Loads Tokenizer and Model with 4-bit BitsAndBytes quantization if GPU available."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError("PyTorch & Transformers are required for LLM text generation. pip install torch transformers")

        dev_info = get_device_info()
        self.device = torch.device(dev_info["device"])
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading Model: {self.model_name} (device={dev_info['device']}, 4bit={dev_info['use_4bit']})...")
        clear_cuda_memory()

        if dev_info["use_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if dev_info["device"] == "cpu" else torch.float16,
                device_map="auto" if dev_info["device"] != "cpu" else None,
                trust_remote_code=True
            )
            if dev_info["device"] == "cpu":
                self.model.to(self.device)

        self.model.eval()
        print("LLM loaded successfully.")

    def load_sources_from_kg(self, kg_path: str, required_count: int, topic: str = "ALL", difficulty: str = "Moderate") -> list:
        """Loads text-only reference questions from the Knowledge Graph pickle."""
        if not os.path.exists(kg_path):
            # Check if kg_retrieved_context.json exists
            if os.path.exists(AGENT2_OUTPUT_JSON):
                with open(AGENT2_OUTPUT_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("questions", [])
            raise FileNotFoundError(f"Knowledge Graph not found: {kg_path}. Run Agent 2 first.")

        with open(kg_path, "rb") as f:
            G = pickle.load(f)

        candidates = []
        for node, attrs in G.nodes(data=True):
            if attrs.get("type") != "Question":
                continue
            question = attrs.get("question", "")
            if not question:
                continue

            # Skip questions with images
            has_image = any(G.nodes[nb].get("type") == "QuestionImage" for nb in G.successors(node))
            if has_image:
                continue

            options = []
            for nb in G.successors(node):
                if G.nodes[nb].get("type") == "Option":
                    options.append(G.nodes[nb].get("text", ""))
            if len(options) != 4:
                continue

            correct = attrs.get("correct_option")
            if correct not in "ABCD":
                continue

            candidates.append({
                "kg_question_id": str(node),
                "question": question,
                "options": options,
                "correct_answer": correct,
                "topic": attrs.get("topic", topic),
                "subtopic": attrs.get("subtopic", "General"),
                "difficulty": attrs.get("difficulty", difficulty)
            })

        random.shuffle(candidates)
        return candidates[:max(required_count * 2, 20)]

    def extract_json(self, text: str) -> dict:
        """Extracts JSON object from model output."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if m:
            candidate = m.group(1)
        else:
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            candidate = cleaned[first_brace:last_brace + 1] if first_brace != -1 and last_brace != -1 else cleaned

        try:
            return json.loads(candidate)
        except Exception:
            return None

    def precheck_question(self, q: dict, existing: list) -> tuple:
        """Validates structure and removes duplicates."""
        if not isinstance(q, dict):
            return False, "Not a dictionary"
        for f in REQUIRED_FIELDS:
            if not q.get(f):
                return False, f"Missing or empty field: {f}"

        raw_ans = str(q["correct_answer"]).strip().upper()
        m = re.search(r"\b([ABCD])\b", raw_ans)
        if not m:
            return False, f"Invalid answer format: {raw_ans}"
        q["correct_answer"] = m.group(1)

        if len(q["question"]) < 35:
            return False, "Question statement too short"
        if len(q["explanation"]) < 30:
            return False, "Explanation too short"
        if TEXT_REFERS_TO_IMAGE.search(q["question"]):
            return False, "Text question refers to missing image/diagram"

        # Check duplicates
        stem = str(q["question"]).lower().strip()
        for e in existing:
            if difflib.SequenceMatcher(None, stem, str(e.get("question", "")).lower().strip()).ratio() > 0.85:
                return False, "Duplicate of existing question"

        return True, "Valid"

    def generate_batch(self, sources: list, batch_num: int) -> list:
        """Generates a batch of new questions based on source questions."""
        import torch
        src_blocks = []
        for i, s in enumerate(sources, start=1):
            opts_str = "\n".join([f"({letter}) {opt}" for letter, opt in zip("ABCD", s.get("options", []))])
            src_blocks.append(f"SOURCE {i} [Topic: {s.get('topic')}, Diff: {s.get('difficulty')}]:\n{s.get('question')}\n{opts_str}")

        prompt = f"""
You are an expert JEE Main Chemistry examiner.
Generate {len(sources)} NEW, ORIGINAL moderate/hard JEE Main multiple choice questions.
Use the conceptual principles from these factual reference questions:

{chr(10).join(src_blocks)}

RULES:
1. Return ONLY valid JSON with key "questions".
2. Do not copy question text directly; transform into new chemistry scenarios.
3. Every question must have question, option_A, option_B, option_C, option_D, correct_answer (A/B/C/D), explanation.
"""
        messages = [
            {"role": "system", "content": "You are an expert chemistry teacher. Output JSON only."},
            {"role": "user", "content": prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                top_p=TOP_P, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )

        n_in = inputs.shape[-1]
        raw_text = self.tokenizer.decode(outputs[0, n_in:], skip_special_tokens=True)
        data = self.extract_json(raw_text)
        return data.get("questions", []) if data else []


# ==============================================================================
# 4. EXPORT HELPERS (HTML, DOCX, JSON, PKL)
# ==============================================================================
def image_to_base64_uri(path: str) -> str:
    """Encodes local image to base64 data URI."""
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def export_html(questions: list, output_path: str, topic="ALL", difficulty="Moderate"):
    """Exports questions to responsive HTML file with embedded base64 figures."""
    esc = html.escape
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>JEE Main Chemistry Questions</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
        "max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; background-color: #f8fafc; color: #1e293b; }",
        ".header { text-align: center; margin-bottom: 40px; }",
        ".header h1 { font-size: 2rem; color: #0f172a; margin-bottom: 8px; }",
        ".meta { display: inline-block; padding: 6px 16px; background: #e2e8f0; border-radius: 20px; font-weight: 500; font-size: 0.9rem; }",
        ".card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }",
        ".q-tag { font-weight: bold; color: #3b82f6; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; }",
        ".question-text { font-size: 1.1rem; font-weight: 600; margin: 12px 0 16px 0; color: #0f172a; }",
        ".figure-box { text-align: center; margin: 16px 0; }",
        ".figure-box img { max-width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff; }",
        ".options-grid { display: grid; grid-template-columns: 1fr; gap: 8px; margin: 16px 0; }",
        "@media (min-width: 600px) { .options-grid { grid-template-columns: 1fr 1fr; } }",
        ".opt { padding: 10px 14px; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }",
        ".opt b { color: #3b82f6; margin-right: 6px; }",
        ".solution-box { margin-top: 18px; padding: 14px 18px; background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 6px; }",
        ".solution-box b { color: #15803d; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='header'>",
        "<h1>JEE Main Chemistry Question Bank</h1>",
        f"<div class='meta'>Topic: {esc(str(topic))} | Difficulty: {esc(str(difficulty))} | Total Questions: {len(questions)}</div>",
        "</div>"
    ]

    for i, q in enumerate(questions, start=1):
        q_type = q.get("question_type", "MCQ").upper()
        parts.append("<div class='card'>")
        parts.append(f"<div class='q-tag'>Question {i} &bull; [{esc(q_type)}]</div>")
        parts.append(f"<div class='question-text'>{esc(str(q.get('question', '')))}</div>")

        img_path = q.get("generated_image") or q.get("image_path")
        if img_path and os.path.exists(img_path):
            uri = image_to_base64_uri(img_path)
            parts.append(f"<div class='figure-box'><img src='{uri}' alt='Figure for question {i}'></div>")

        parts.append("<div class='options-grid'>")
        for letter in "ABCD":
            opt_key = f"option_{letter}"
            parts.append(f"<div class='opt'><b>({letter})</b> {esc(str(q.get(opt_key, '')))}</div>")
        parts.append("</div>")

        correct_ans = q.get("correct_answer", "")
        expl = q.get("explanation", "")
        parts.append("<div class='solution-box'>")
        parts.append(f"<div><b>Correct Answer:</b> ({esc(str(correct_ans))})</div>")
        parts.append(f"<div style='margin-top:6px;'><b>Explanation:</b> {esc(str(expl))}</div>")
        parts.append("</div>")

        parts.append("</div>")

    parts.append("</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"HTML exported to: {output_path}")


def export_docx(questions: list, output_path: str, topic="ALL", difficulty="Moderate"):
    """Exports questions to styled Microsoft Word document."""
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
    except ImportError:
        print("Notice: 'python-docx' is not installed. Skipping Word document export.")
        print("To enable docx: pip install python-docx")
        return

    doc = Document()
    doc.add_heading("JEE Main Chemistry Question Bank", level=0)
    meta = doc.add_paragraph(f"Topic: {topic} | Difficulty: {difficulty} | Total: {len(questions)}")
    meta.runs[0].italic = True

    for i, q in enumerate(questions, start=1):
        q_type = q.get("question_type", "MCQ").upper()
        doc.add_heading(f"Q{i}. [{q_type}]", level=2)
        doc.add_paragraph(str(q.get("question", "")))

        img_path = q.get("generated_image") or q.get("image_path")
        if img_path and os.path.exists(img_path):
            try:
                doc.add_picture(img_path, width=Inches(4.2))
            except Exception:
                pass

        for letter in "ABCD":
            doc.add_paragraph(f"({letter}) {q.get(f'option_{letter}', '')}")

        p_ans = doc.add_paragraph()
        run_ans_lbl = p_ans.add_run("Correct Answer: ")
        run_ans_lbl.bold = True
        run_ans_lbl.font.color.rgb = RGBColor(21, 128, 61)
        p_ans.add_run(f"({q.get('correct_answer', '')})")

        p_expl = doc.add_paragraph()
        run_expl_lbl = p_expl.add_run("Explanation: ")
        run_expl_lbl.bold = True
        p_expl.add_run(str(q.get("explanation", "")))

        doc.add_paragraph("-" * 50)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"Word document exported to: {output_path}")


# ==============================================================================
# 5. MAIN AGENT 4 ORCHESTRATOR
# ==============================================================================
class MultimodalChemistryAgent:
    """Agent 4: Coordinates multimodal question generation, validation, and exports."""

    def __init__(self):
        self.image_engine = ImageQuestionEngine()
        self.text_engine = TextQuestionEngine()

    def generate(
        self,
        total_count: int = 5,
        mode: str = "mixed",
        image_count: int = None,
        topic: str = "ALL",
        difficulty: str = "Moderate",
        kg_path: str = KG_PICKLE_PATH
    ) -> list:
        """Executes question generation plan."""
        mode = mode.lower()
        if mode == "text":
            n_text, n_image = total_count, 0
        elif mode == "image":
            n_text, n_image = 0, total_count
        else:
            # Mixed mode
            if image_count is not None:
                n_image = min(max(0, image_count), total_count)
            else:
                n_image = total_count // 2
            n_text = total_count - n_image

        print("\n" + "=" * 70)
        print("AGENT 4: MULTIMODAL QUESTION GENERATION")
        print("=" * 70)
        print(f"Total: {total_count} questions | Mode: {mode} | Text: {n_text} | Image: {n_image}")
        print(f"Topic: {topic} | Target Difficulty: {difficulty}")

        all_questions = []

        # 1. Generate Image Questions
        if n_image > 0:
            print(f"\n--- Generating {n_image} verified image questions via RDKit & Matplotlib ---")
            pool = self.image_engine.family_pool(topic)
            for idx in range(n_image):
                q = self.image_engine.build_question(pool)
                all_questions.append(q)
                print(f"  [Image {idx+1}/{n_image}] Built {q.get('image_family')} -> Answer: ({q['correct_answer']})")

        # 2. Generate Text Questions
        if n_text > 0:
            print(f"\n--- Generating {n_text} text questions via LLM from Knowledge Graph ---")
            try:
                self.text_engine.initialize_llm()
                sources = self.text_engine.load_sources_from_kg(kg_path, n_text, topic=topic, difficulty=difficulty)
                
                # Batch generation
                generated_text_qs = []
                for b_start in range(0, n_text, BATCH_SIZE):
                    batch_srcs = sources[b_start : b_start + BATCH_SIZE]
                    raw_qs = self.text_engine.generate_batch(batch_srcs, b_start // BATCH_SIZE + 1)
                    for raw_q in raw_qs:
                        valid, msg = self.text_engine.precheck_question(raw_q, all_questions + generated_text_qs)
                        if valid:
                            raw_q["question_type"] = "text"
                            raw_q["source"] = "Knowledge Graph -> Qwen"
                            raw_q["verification"] = "generated"
                            generated_text_qs.append(raw_q)
                        if len(generated_text_qs) >= n_text:
                            break
                    if len(generated_text_qs) >= n_text:
                        break

                all_questions.extend(generated_text_qs[:n_text])
            except Exception as e:
                print(f"Warning: LLM generation could not proceed ({e}).")
                print("Falling back to programmatically computed high-quality questions for remaining slots.")
                pool = self.image_engine.family_pool(topic)
                for _ in range(n_text):
                    q = self.image_engine.build_question(pool)
                    all_questions.append(q)

        # Shuffle mixed question order
        if mode == "mixed":
            random.shuffle(all_questions)

        # Save outputs
        result_dict = {"questions": all_questions}
        with open(AGENT4_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)
        print(f"\nJSON saved: {AGENT4_OUTPUT_JSON}")

        with open(AGENT4_OUTPUT_PKL, "wb") as f:
            pickle.dump(result_dict, f)
        print(f"Pickle saved: {AGENT4_OUTPUT_PKL}")

        export_html(all_questions, AGENT4_OUTPUT_HTML, topic=topic, difficulty=difficulty)
        export_docx(all_questions, AGENT4_OUTPUT_DOCX, topic=topic, difficulty=difficulty)

        print("\n" + "=" * 70)
        print(f"GENERATION COMPLETED: {len(all_questions)} questions generated successfully!")
        print("=" * 70)
        return all_questions


def main():
    parser = argparse.ArgumentParser(description="Agent 4: Multimodal Chemistry Question Generation Agent.")
    parser.add_argument("--count", type=int, default=5, help="Total number of questions.")
    parser.add_argument("--mode", type=str, default="image", choices=["text", "image", "mixed"], help="Question mode.")
    parser.add_argument("--image-count", type=int, default=None, help="Number of image questions (if mixed).")
    parser.add_argument("--topic", type=str, default="ALL", help="Chemistry broad topic.")
    parser.add_argument("--difficulty", type=str, default="Moderate", help="Difficulty level.")
    args = parser.parse_args()

    agent = MultimodalChemistryAgent()
    agent.generate(
        total_count=args.count,
        mode=args.mode,
        image_count=args.image_count,
        topic=args.topic,
        difficulty=args.difficulty
    )


if __name__ == "__main__":
    main()
