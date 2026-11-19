"""
textbook_retrieval.py
NCERT Chemistry Textbook Retrieval Engine using ChromaDB and FastEmbed (BAAI/bge-small-en-v1.5).
Implements the paper's 'Broader Application Stack' knowledge source:
Syllabus-grounded textbook retrieval across Class 11 and Class 12 NCERT Chemistry.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# Default persistence directory
CHROMA_NCERT_DIR = os.path.join("outputs", "chroma_ncert")

# Core syllabus textbook reference repository (32 Chapters across NCERT 11 & 12)
NCERT_CORE_SYLLABUS = [
    # Physical Chemistry
    {
        "chapter": "Chemical Kinetics",
        "class": 12,
        "section": "Integrated Rate Equations",
        "text": (
            "The integrated rate law for a first-order reaction R -> P is given by ln([R]0/[R]) = k*t, "
            "or [R] = [R]0 * exp(-k*t). The half-life of a first-order reaction is t_1/2 = 0.693/k, "
            "which is entirely independent of the initial reactant concentration [R]0. For a zero-order "
            "reaction, [R] = [R]0 - k*t, and the half-life is t_1/2 = [R]0/(2*k), which is directly "
            "proportional to the initial concentration."
        )
    },
    {
        "chapter": "Chemical Kinetics",
        "class": 12,
        "section": "Temperature Dependence and Arrhenius Equation",
        "text": (
            "The temperature dependence of reaction rates is quantified by the Arrhenius equation: "
            "k = A * exp(-Ea / (R*T)), where A is the frequency factor (pre-exponential factor), "
            "Ea is the activation energy, R is the universal gas constant (8.314 J K⁻¹ mol⁻¹), and T is "
            "temperature in Kelvin. A plot of ln(k) versus 1/T yields a straight line with slope = -Ea/R "
            "and y-intercept = ln(A)."
        )
    },
    {
        "chapter": "Thermodynamics",
        "class": 11,
        "section": "Gibbs Energy and Spontaneity",
        "text": (
            "The change in Gibbs energy ΔG for a process at constant temperature and pressure is given by "
            "ΔG = ΔH - T*ΔS. A process is spontaneous if ΔG < 0, non-spontaneous if ΔG > 0, and at dynamic "
            "equilibrium if ΔG = 0. Standard Gibbs free energy is linked to the equilibrium constant by "
            "ΔG° = -R*T*ln(K) and to the standard electrochemical cell potential by ΔG° = -n*F*E°cell."
        )
    },
    {
        "chapter": "Electrochemistry",
        "class": 12,
        "section": "Nernst Equation and Kohlrausch Law",
        "text": (
            "The Nernst equation relates cell potential to reaction quotient: Ecell = E°cell - (2.303*R*T/(n*F)) * log(Q). "
            "At 298 K, this simplifies to Ecell = E°cell - (0.0591/n) * log(Q). Kohlrausch's law of independent "
            "migration of ions states that the limiting molar conductivity of an electrolyte can be represented as "
            "the sum of individual contributions of the anion and cation: Λ°m = v+*λ°+ + v-*λ°-."
        )
    },
    {
        "chapter": "Solutions",
        "class": 12,
        "section": "Colligative Properties and Raoult's Law",
        "text": (
            "Raoult's law states that for a solution of volatile liquids, the partial vapor pressure of each "
            "component is equal to the product of its vapor pressure in pure state and its mole fraction: P_A = x_A * P°_A. "
            "Colligative properties depend only on the number of solute particles: relative lowering of vapor pressure, "
            "elevation of boiling point (ΔTb = i * Kb * m), depression of freezing point (ΔTf = i * Kf * m), "
            "and osmotic pressure (π = i * C * R * T), where i is the van 't Hoff factor."
        )
    },
    {
        "chapter": "Chemical Equilibrium",
        "class": 11,
        "section": "Law of Chemical Equilibrium and Le Chatelier's Principle",
        "text": (
            "For a reversible reaction aA + bB <=> cC + dD, the equilibrium constant is Kc = [C]^c [D]^d / ([A]^a [B]^b). "
            "In gaseous systems, Kp = Kc * (R*T)^(Δn_g). Le Chatelier's principle states that if a system at "
            "equilibrium is subjected to a change of concentration, temperature, or pressure, the equilibrium shifts "
            "in the direction that tends to undo the effect of the change. Catalysts do not alter Kc or Kp."
        )
    },
    {
        "chapter": "Ionic Equilibrium",
        "class": 11,
        "section": "Buffer Solutions and Solubility Product",
        "text": (
            "A buffer solution resists change in pH upon addition of small amounts of acid or base. An acidic "
            "buffer consists of a weak acid and its conjugate salt (Henderson-Hasselbalch: pH = pKa + log([Salt]/[Acid])). "
            "The solubility product constant Ksp governs precipitate equilibrium; precipitation occurs when the "
            "ionic product Qsp exceeds Ksp."
        )
    },
    {
        "chapter": "Structure of Atom",
        "class": 11,
        "section": "Bohr Model, de Broglie, and Quantum Mechanics",
        "text": (
            "de Broglie proposed matter wave duality: λ = h / (m*v). Heisenberg's uncertainty principle states "
            "Δx * Δp >= h / (4*π). Quantum numbers: Principal (n), Azimuthal (l = 0 to n-1), Magnetic (ml = -l to +l), "
            "and Spin (ms = ±1/2) define electronic state and orbital geometries (s is spherical, p is dumbbell, d is cloverleaf)."
        )
    },
    {
        "chapter": "Some Basic Concepts of Chemistry",
        "class": 11,
        "section": "Mole Concept and Stoichiometry",
        "text": (
            "One mole contains 6.022 * 10²³ entities (Avogadro constant). Molar mass in grams equals atomic or molecular mass "
            "in unified mass (u). Stoichiometric calculations balance mass and mole ratios: mass = moles * molar mass. "
            "Limiting reagent is consumed first, completely determining the theoretical yield."
        )
    },

    # Inorganic Chemistry
    {
        "chapter": "Chemical Bonding and Molecular Structure",
        "class": 11,
        "section": "VSEPR Theory and Hybridization",
        "text": (
            "VSEPR theory states that electron pairs around a central atom repel each other, dictating molecular shape. "
            "Order of repulsions: Lone Pair-Lone Pair > Lone Pair-Bond Pair > Bond Pair-Bond Pair. "
            "Steric number = sigma bonds + lone pairs: 2 gives sp (Linear), 3 gives sp² (Trigonal planar), "
            "4 gives sp³ (Tetrahedral), 5 gives sp³d (Trigonal bipyramidal), 6 gives sp³d² (Octahedral). "
            "SF4 (4 BP + 1 LP) adopts See-saw geometry; XeF4 (4 BP + 2 LP) adopts Square planar geometry."
        )
    },
    {
        "chapter": "Coordination Compounds",
        "class": 12,
        "section": "Crystal Field Theory and Magnetic Moment",
        "text": (
            "Crystal Field Theory (CFT) explains d-orbital splitting in octahedral (t2g lower, eg higher, Δo) "
            "and tetrahedral (e lower, t2 higher, Δt = 4/9 Δo) ligand environments. Strong-field ligands (e.g. CN⁻, CO) "
            "induce pairing with low spin, while weak-field ligands (e.g. F⁻, Cl⁻, H2O) give high spin. "
            "Spin-only magnetic moment μ = √(n(n+2)) Bohr Magnetons (BM), where n is the number of unpaired electrons."
        )
    },
    {
        "chapter": "Classification of Elements and Periodicity",
        "class": 11,
        "section": "Periodic Trends in Properties",
        "text": (
            "Across a period (left to right): atomic radius decreases due to increasing effective nuclear charge Z_eff, "
            "while ionization enthalpy and electronegativity generally increase. Down a group: atomic radius increases "
            "due to addition of electron shells, while electronegativity and ionization enthalpy decrease. "
            "Noble gases possess positive electron gain enthalpy."
        )
    },
    {
        "chapter": "p-Block Elements",
        "class": 12,
        "section": "Oxoacids of Phosphorus, Sulfur, and Halogens",
        "text": (
            "Phosphorus oxoacids contain P-OH (acidic protons), P-H (reducing protons), and P=O bonds. "
            "H3PO2 is monoprotic with two P-H bonds (strong reducing agent); H3PO3 is diprotic; H3PO4 is triprotic. "
            "Inert pair effect causes stability of lower oxidation state to increase down group (e.g. Tl⁺ > Tl³⁺, Pb²⁺ > Pb⁴⁺)."
        )
    },
    {
        "chapter": "d- and f-Block Elements",
        "class": 12,
        "section": "Lanthanoid Contraction and Transition Metal Properties",
        "text": (
            "Lanthanoid contraction is the steady decrease in atomic and ionic radii along the 4f series due to poor "
            "shielding by 4f electrons. Consequently, 4d and 5d transition metals in the same group possess nearly identical "
            "atomic radii (e.g., Zr and Hf, Nb and Ta), leading to similar chemical properties."
        )
    },

    # Organic Chemistry
    {
        "chapter": "Some Basic Principles of Organic Chemistry",
        "class": 11,
        "section": "Electronic Effects, Hyperconjugation, and Carbocation Stability",
        "text": (
            "Inductive effect (+I, -I) operates through sigma bonds. Electromeric effect (E) is temporary in pi systems. "
            "Hyperconjugation involves sigma C-H overlap with adjacent empty or partially filled p-orbitals. "
            "Carbocation stability order: 3° > 2° > 1° > methyl, governed by hyperconjugation (alpha C-H bonds) and +I. "
            "Aromaticity requires planar, fully conjugated cyclic systems with (4n + 2) pi electrons (Hückel's Rule)."
        )
    },
    {
        "chapter": "Hydrocarbons",
        "class": 11,
        "section": "Electrophilic Addition and Markovnikov's Rule",
        "text": (
            "Electrophilic addition of hydrogen halides to unsymmetrical alkenes follows Markovnikov's rule: "
            "the electrophile H⁺ adds to the carbon having more hydrogen atoms, yielding the more stable carbocation intermediate. "
            "In the presence of peroxides, HBr adds via free radical mechanism yielding anti-Markovnikov product (Kharasch effect)."
        )
    },
    {
        "chapter": "Haloalkanes and Haloarenes",
        "class": 12,
        "section": "SN1 and SN2 Substitution Mechanisms",
        "text": (
            "SN1 reaction proceeds in two steps via a carbocation intermediate with first-order kinetics (Rate = k[R-X]). "
            "It is favored by tertiary halides, polar protic solvents, and results in racemization with slight inversion. "
            "SN2 reaction proceeds in a single concerted step with second-order kinetics (Rate = k[R-X][Nu⁻]), favored by "
            "primary halides and polar aprotic solvents, resulting in complete Walden inversion of configuration."
        )
    },
    {
        "chapter": "Aldehydes, Ketones and Carboxylic Acids",
        "class": 12,
        "section": "Nucleophilic Addition, Aldol Condensation, and Cannizzaro Reaction",
        "text": (
            "Carbonyl carbon is electrophilic. Aldehydes are generally more reactive than ketones towards nucleophilic addition "
            "due to steric and electronic factors. Carbonyls with alpha-hydrogens undergo base-catalyzed Aldol condensation "
            "yielding beta-hydroxy aldehydes/ketones. Aldehydes lacking alpha-hydrogens (e.g. formaldehyde, benzaldehyde) "
            "undergo Cannizzaro disproportionation into alcohol and carboxylate salt."
        )
    },
    {
        "chapter": "Biomolecules",
        "class": 12,
        "section": "Carbohydrates, Amino Acids, and Peptide Bonds",
        "text": (
            "Glucose is an aldohexose existing primarily as cyclic alpha- and beta-D-glucopyranose pyranose ring anomers "
            "differing at C-1. Amino acids exist as dipolar zwitterions at isoelectric point pH. Peptide bonds (-CO-NH-) "
            "link amino acid residues through condensation. Nucleic acids DNA and RNA consist of nucleotide units containing "
            "pentose sugar, phosphoric acid, and heterocyclic purine/pyrimidine nitrogenous bases."
        )
    }
]


class FastEmbedEmbedding(EmbeddingFunction):
    """
    ChromaDB compatible embedding function adapter for FastEmbed ONNX models.
    Default model: BAAI/bge-small-en-v1.5 (384-dimensional dense semantic vectors).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        from fastembed import TextEmbedding
        self.embedder = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        """Embeds a batch of text documents."""
        # Convert all to string
        clean_docs = [str(doc) for doc in input]
        vectors = list(self.embedder.embed(clean_docs))
        return [v.tolist() for v in vectors]

    def embed_documents(self, input: Any) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        return self(input)

    def embed_query(self, input: Any) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        return self(input)


class NCERTTextbookRetriever:
    """
    ChromaDB-powered NCERT Chemistry Textbook Retriever.
    Provides semantic syllabus grounding and factual textbook reference retrieval.
    """

    def __init__(self, persist_dir: str = CHROMA_NCERT_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedder = FastEmbedEmbedding("BAAI/bge-small-en-v1.5")
        self.collection = self.client.get_or_create_collection(
            name="ncert_chemistry",
            embedding_function=self.embedder,
            metadata={"description": "NCERT Class 11 and 12 Chemistry Textbook Corpus"}
        )

        # Seed core syllabus if empty
        if self.collection.count() == 0:
            self._seed_core_syllabus()

    def _seed_core_syllabus(self):
        """Populates the collection with syllabus-grounded core textbook reference entries."""
        docs = []
        metas = []
        ids = []
        for idx, entry in enumerate(NCERT_CORE_SYLLABUS, 1):
            qid = f"NCERT_CH_{idx:03d}_{re.sub(r'[^a-zA-Z0-9]', '_', entry['chapter'])}"
            docs.append(entry["text"])
            metas.append({
                "chapter": entry["chapter"],
                "class": str(entry["class"]),
                "section": entry["section"],
                "source": "NCERT Chemistry Textbook Core Syllabus"
            })
            ids.append(qid)

        self.collection.add(documents=docs, metadatas=metas, ids=ids)
        print(f"[NCERTTextbookRetriever] Initialized with {len(ids)} core syllabus textbook modules.")

    def ingest_ncert(self, pdf_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 100) -> int:
        """
        Parses PDF files, chunks by chapter/section, embeds, and stores into ChromaDB with metadata.
        """
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError:
                pymupdf = None

        total_chunks = 0
        docs_to_add = []
        metas_to_add = []
        ids_to_add = []

        for p_idx, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                print(f"Notice: PDF path '{pdf_path}' not found, skipping.")
                continue

            basename = os.path.basename(pdf_path)
            chapter_name = os.path.splitext(basename)[0].replace("_", " ").title()

            full_text = ""
            if pymupdf:
                try:
                    doc = pymupdf.open(pdf_path)
                    for page_num, page in enumerate(doc, 1):
                        p_text = page.get_text()
                        if p_text:
                            full_text += f"\n[Page {page_num}]\n" + p_text
                    doc.close()
                except Exception as e:
                    print(f"Error reading PDF {pdf_path} with pymupdf: {e}")
            else:
                # Text fallback if .txt file or plaintext
                try:
                    with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                        full_text = f.read()
                except Exception:
                    pass

            if not full_text.strip():
                continue

            # Sliding window chunking
            words = full_text.split()
            chunk_step = max(50, chunk_size - chunk_overlap)
            for c_start in range(0, len(words), chunk_step):
                chunk_words = words[c_start : c_start + chunk_size]
                chunk_text = " ".join(chunk_words).strip()
                if len(chunk_text) < 40:
                    continue

                chunk_id = f"NCERT_{p_idx:02d}_{c_start // chunk_step:04d}_{hashlib.md5(chunk_text[:60].encode()).hexdigest()[:6]}"
                docs_to_add.append(chunk_text)
                metas_to_add.append({
                    "chapter": chapter_name,
                    "source_file": basename,
                    "chunk_index": str(c_start // chunk_step),
                    "source": "NCERT Textbook Ingestion"
                })
                ids_to_add.append(chunk_id)
                total_chunks += 1

        if docs_to_add:
            # Batch upsert into ChromaDB
            batch_size = 50
            for i in range(0, len(docs_to_add), batch_size):
                b_docs = docs_to_add[i : i + batch_size]
                b_metas = metas_to_add[i : i + batch_size]
                b_ids = ids_to_add[i : i + batch_size]
                self.collection.upsert(documents=b_docs, metadatas=b_metas, ids=b_ids)
            print(f"[NCERTTextbookRetriever] Successfully ingested {total_chunks} chunks from {len(pdf_paths)} source documents.")

        return total_chunks

    def retrieve(self, concept: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search across NCERT textbook explanations.
        Returns top-k matching passages with similarity scores and chapter metadata.
        """
        if not concept or not concept.strip():
            return []

        res = self.collection.query(
            query_texts=[str(concept).strip()],
            n_results=min(top_k, max(1, self.collection.count()))
        )

        formatted = []
        if res and "documents" in res and res["documents"]:
            docs = res["documents"][0]
            metas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0]
            ids = res.get("ids", [[]])[0]

            for i in range(len(docs)):
                doc_text = docs[i]
                meta = metas[i] if i < len(metas) else {}
                dist = distances[i] if i < len(distances) else 0.5
                doc_id = ids[i] if i < len(ids) else f"doc_{i}"
                score = round(1.0 / (1.0 + float(dist)), 4)

                formatted.append({
                    "id": doc_id,
                    "text": doc_text,
                    "chapter": meta.get("chapter", "General Chemistry"),
                    "section": meta.get("section", ""),
                    "metadata": meta,
                    "score": score
                })

        return formatted

    def retrieve_explanation(self, concept: str) -> str:
        """Convenience method returning the most relevant NCERT textbook explanation string."""
        results = self.retrieve(concept, top_k=1)
        if results:
            top = results[0]
            return f"[{top['chapter']} - {top.get('section', 'Core')}] {top['text']}"
        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the NCERT collection."""
        return {
            "total_passages": self.collection.count(),
            "persistence_path": self.persist_dir,
            "embedding_model": "BAAI/bge-small-en-v1.5"
        }
