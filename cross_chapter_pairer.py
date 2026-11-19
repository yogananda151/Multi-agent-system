"""
cross_chapter_pairer.py
Systematic Cross-Chapter Pairing Engine via Knowledge Graph Traversal (CD-NS-KG-MAS Eq. 2).
Finds chemically coupled concept pairs across chapters for Hard questions.
Formula: C = {(u, v) in E | chapter(u) != chapter(v)}
"""

import os
import re
import pickle
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple

from config import KG_PICKLE_PATH, CROSS_CHAPTER_PAIRS


class CrossChapterPairer:
    """Finds chemically coupled concept pairs across chapters for Hard questions."""

    # Extended Cross-Chapter Coupling Governing Principles Database
    COUPLING_DB = {
        ("gibbs free energy", "standard cell potential"): {
            "pair_id": "X_THERMO_ELECTRO",
            "formula": "ΔG° = -nFE°cell",
            "description": "Thermodynamic spontaneity ↔ electrochemical driving force"
        },
        ("arrhenius activation energy", "equilibrium constant"): {
            "pair_id": "X_KINETICS_EQUILIBRIUM",
            "formula": "d(ln K)/dT = ΔH°/RT²; k = Ae^(-Ea/RT)",
            "description": "Temperature dependence of rates ↔ dynamic equilibrium shift"
        },
        ("carbocation stability", "sn1/sn2 kinetics"): {
            "pair_id": "X_GOC_HALOALKANES",
            "formula": "Rate = k[R-X] (SN1) vs Rate = k[R-X][Nu⁻] (SN2)",
            "description": "Carbocation hyperconjugation ↔ substitution kinetics & stereochemistry"
        },
        ("vsepr geometry", "crystal field splitting"): {
            "pair_id": "X_BONDING_COORDINATION",
            "formula": "μ_eff = √(n(n+2)) BM; Δ_o = f(ligand field)",
            "description": "Hybridization geometry ↔ d-orbital splitting and magnetic moments"
        },
        ("colligative properties", "entropy of mixing"): {
            "pair_id": "X_SOLUTIONS_THERMO",
            "formula": "ΔT_f = i * K_f * m; ΔG_mix = ΔH_mix - T*ΔS_mix",
            "description": "Thermodynamic non-ideality ↔ freezing point depression"
        },
        ("hess's law enthalpy", "reaction kinetics"): {
            "pair_id": "X_THERMO_KINETICS",
            "formula": "ΔH_reaction = Ea(forward) - Ea(reverse)",
            "description": "Thermochemical enthalpy cycle ↔ forward/reverse activation energy barriers"
        },
        ("faraday's electrolysis", "mole concept stoichiometry"): {
            "pair_id": "X_ELECTRO_STOICHIOMETRY",
            "formula": "w = (M * I * t) / (n * F)",
            "description": "Charge transfer Faraday law ↔ stoichiometric equivalent mass"
        },
        ("kohlrausch conductivity", "ostwald dilution law"): {
            "pair_id": "X_CONDUCTANCE_IONIC",
            "formula": "α = Λ_m / Λ°_m; K_a = (c * α²) / (1 - α)",
            "description": "Limiting molar ionic mobility ↔ weak electrolyte dissociation equilibrium"
        },
        ("raoult's vapor pressure", "gas phase equilibrium"): {
            "pair_id": "X_SOLUTIONS_EQUILIBRIUM",
            "formula": "P_A = X_A * P°_A; K_p = P_B / P_A",
            "description": "Liquid-vapor mole fraction equilibrium ↔ partial pressure equilibrium constant"
        },
        ("de broglie wavelength", "photoelectric effect"): {
            "pair_id": "X_ATOMIC_MODERN",
            "formula": "λ = h / (m * v); hν = Φ + (1/2)m*v²_max",
            "description": "Matter wave duality ↔ threshold work function and photon momentum"
        }
    }

    CHAPTER_MAP = {
        "chemical kinetics": "Chemical Kinetics",
        "equilibrium": "Equilibrium",
        "chemical equilibrium": "Equilibrium",
        "ionic equilibrium": "Equilibrium",
        "thermodynamics": "Thermodynamics",
        "electrochemistry": "Electrochemistry",
        "general organic chemistry": "General Organic Chemistry",
        "hydrocarbon": "General Organic Chemistry",
        "haloalkanes": "Haloalkanes and Haloarenes",
        "haloalkanes and haloarenes": "Haloalkanes and Haloarenes",
        "chemical bonding": "Chemical Bonding and Molecular Structure",
        "coordination compounds": "Coordination Compounds",
        "solutions": "Solutions",
        "atomic structure": "Atomic Structure",
        "mole concept": "Some Basic Concepts of Chemistry"
    }

    def __init__(self, kg_path: str = KG_PICKLE_PATH):
        self.kg_path = kg_path
        self.kg = None
        if os.path.exists(self.kg_path):
            try:
                with open(self.kg_path, "rb") as f:
                    self.kg = pickle.load(f)
            except Exception:
                self.kg = None

        if self.kg is None:
            # Fallback to symbolic ontological graph
            self.kg = self._build_ontological_graph()

    def _build_ontological_graph(self) -> nx.DiGraph:
        """Constructs a lightweight ontological knowledge graph linking chapters and concepts."""
        G = nx.DiGraph()
        for pair in CROSS_CHAPTER_PAIRS:
            chA = pair["chapter_A"]
            chB = pair["chapter_B"]
            cA = pair["concept_A"]
            cB = pair["concept_B"]

            G.add_node(f"TOPIC_{chA}", type="Topic", label=chA)
            G.add_node(f"TOPIC_{chB}", type="Topic", label=chB)
            G.add_node(f"CONCEPT_{cA}", type="Concept", label=cA, chapter=chA)
            G.add_node(f"CONCEPT_{cB}", type="Concept", label=cB, chapter=chB)

            G.add_edge(f"TOPIC_{chA}", f"CONCEPT_{cA}", relation="HAS_CONCEPT")
            G.add_edge(f"TOPIC_{chB}", f"CONCEPT_{cB}", relation="HAS_CONCEPT")
            G.add_edge(f"CONCEPT_{cA}", f"CONCEPT_{cB}", relation="CROSS_COUPLED",
                       formula=pair["coupling_formula"], description=pair["description"])

        return G

    def _get_concepts_under_topic(self, target_topic: str) -> List[str]:
        """Traverses the KG to locate concept entities linked to target topic."""
        t_norm = str(target_topic).strip().lower()
        matched_concepts = []

        # 1. Search in NetworkX graph nodes
        for node, data in self.kg.nodes(data=True):
            ntype = data.get("type", "")
            label = str(data.get("label", "")).strip()

            if ntype in ["Topic", "Subtopic"]:
                if t_norm in label.lower() or label.lower() in t_norm:
                    # Collect connected concepts or subtopics
                    for succ in self.kg.successors(node):
                        sdata = self.kg.nodes.get(succ, {})
                        slabel = sdata.get("label", succ)
                        matched_concepts.append(str(slabel))

        # 2. Add ontological concepts from knowledge base
        for (c1, c2), info in self.COUPLING_DB.items():
            pair_id = info["pair_id"]
            for pair in CROSS_CHAPTER_PAIRS:
                if pair.get("pair_id") == pair_id:
                    if t_norm in pair["chapter_A"].lower() or pair["chapter_A"].lower() in t_norm:
                        matched_concepts.append(pair["concept_A"])
                    if t_norm in pair["chapter_B"].lower() or pair["chapter_B"].lower() in t_norm:
                        matched_concepts.append(pair["concept_B"])

        # Deduplicate
        return list(dict.fromkeys(matched_concepts))

    def _get_related_concepts(self, concept: str) -> List[str]:
        """Finds chemically linked concepts via graph edges and coupling database."""
        c_norm = str(concept).strip().lower()
        related = []

        # Graph edge traversal
        c_node = f"CONCEPT_{concept}"
        if c_node in self.kg:
            for succ in self.kg.successors(c_node):
                related.append(str(self.kg.nodes[succ].get("label", succ)))
            for pred in self.kg.predecessors(c_node):
                related.append(str(self.kg.nodes[pred].get("label", pred)))

        # Coupling DB traversal
        for (c1, c2) in self.COUPLING_DB:
            if c1 in c_norm or c_norm in c1:
                related.append(c2.title())
            elif c2 in c_norm or c_norm in c2:
                related.append(c1.title())

        return list(dict.fromkeys(related))

    def _get_chapter(self, concept: str) -> str:
        """Resolves the governing JEE syllabus chapter for a given concept."""
        c_norm = str(concept).strip().lower()

        # Check graph node attributes
        c_node = f"CONCEPT_{concept}"
        if c_node in self.kg and "chapter" in self.kg.nodes[c_node]:
            return self.kg.nodes[c_node]["chapter"]

        for key, chapter in self.CHAPTER_MAP.items():
            if key in c_norm:
                return chapter

        # Check existing CROSS_CHAPTER_PAIRS
        for pair in CROSS_CHAPTER_PAIRS:
            if c_norm in pair["concept_A"].lower() or pair["concept_A"].lower() in c_norm:
                return pair["chapter_A"]
            if c_norm in pair["concept_B"].lower() or pair["concept_B"].lower() in c_norm:
                return pair["chapter_B"]

        return "Physical Chemistry"

    def _get_coupling_principle(self, c1: str, c2: str) -> Optional[Dict[str, str]]:
        """Maps concept pair to governing thermodynamic / kinetic formulas."""
        k1 = c1.strip().lower()
        k2 = c2.strip().lower()

        for (db1, db2), info in self.COUPLING_DB.items():
            if (db1 in k1 or k1 in db1) and (db2 in k2 or k2 in db2):
                return info
            if (db1 in k2 or k2 in db1) and (db2 in k1 or k1 in db2):
                return info

        return None

    def find_pairs(self, target_topic: str, max_pairs: int = 3) -> List[Dict[str, Any]]:
        """
        Traverses Knowledge Graph and Coupling DB to find valid cross-chapter couplings
        for the target topic.
        """
        target_concepts = self._get_concepts_under_topic(target_topic)
        pairs = []
        seen_pairs = set()

        for c1 in target_concepts:
            for c2 in self._get_related_concepts(c1):
                ch1 = self._get_chapter(c1)
                ch2 = self._get_chapter(c2)
                if ch1 != ch2:
                    coupling = self._get_coupling_principle(c1, c2)
                    if coupling:
                        pair_key = tuple(sorted([c1.lower(), c2.lower()]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            pairs.append({
                                "pair_id": coupling.get("pair_id", f"X_{ch1[:4].upper()}_{ch2[:4].upper()}"),
                                "concept_A": c1,
                                "concept_B": c2,
                                "chapter_A": ch1,
                                "chapter_B": ch2,
                                "coupling_formula": coupling["formula"],
                                "description": coupling["description"]
                            })
                            if len(pairs) >= max_pairs:
                                return pairs

        # Fallback to topic-matched pairs from CROSS_CHAPTER_PAIRS if traversal yields fewer
        t_low = target_topic.lower()
        for p in CROSS_CHAPTER_PAIRS:
            if t_low in p["chapter_A"].lower() or t_low in p["chapter_B"].lower():
                pair_key = tuple(sorted([p["concept_A"].lower(), p["concept_B"].lower()]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append(p)
                    if len(pairs) >= max_pairs:
                        return pairs

        # Default fallback
        if not pairs and CROSS_CHAPTER_PAIRS:
            pairs.append(CROSS_CHAPTER_PAIRS[0])

        return pairs
