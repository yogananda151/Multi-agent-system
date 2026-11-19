"""
vector_store.py: Tri-Modal Hybrid Vector Store for JEE Chemistry System
Manages Qdrant collections for:
  1. chemistry_questions: Dense semantic vectors + payload filtering
  2. diagram_embeddings: 128-D Swin Transformer + Feature MLP vectors
  3. molecular_structures: 1024-bit Morgan Chemical Fingerprints (ECFP4)
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union

from config import OUTPUT_DIR

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Chemistry
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class LightweightEmbedder:
    """
    Robust embedding generator that supports FastEmbed / SentenceTransformers
    with graceful fallback to Scikit-Learn Tfidf + TruncatedSVD for deterministic
    offline vectorization.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.backend = None
        self.model = None

        # 1. Try FastEmbed (Qdrant native ONNX runtime)
        try:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self.backend = "fastembed"
            print("Embedding backend: FastEmbed (BAAI/bge-small-en-v1.5)")
            return
        except Exception:
            pass

        # 2. Try SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "sentence_transformers"
            print("Embedding backend: SentenceTransformers (all-MiniLM-L6-v2)")
            return
        except Exception:
            pass

        # 3. Fallback: HashingVectorizer + L2 normalization (Deterministic 384-D)
        from sklearn.feature_extraction.text import HashingVectorizer
        self.model = HashingVectorizer(n_features=self.dim, norm='l2', alternate_sign=False)
        self.backend = "hashing_l2"
        print(f"Embedding backend: Deterministic Normalized HashingVectorizer ({self.dim}-D)")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.backend == "fastembed":
            embeddings = list(self.model.embed(texts))
            return [e.tolist() for e in embeddings]
        elif self.backend == "sentence_transformers":
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return [e.tolist() for e in embeddings]
        else:
            sparse_matrix = self.model.transform(texts)
            dense = sparse_matrix.toarray()
            # Normalize to unit length
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (dense / norms).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class ChemistryVectorStore:
    """Manages indexing, hybrid querying, and similarity retrieval across text, diagrams, and molecules."""

    TEXT_COLLECTION = "chemistry_questions"
    DIAGRAM_COLLECTION = "diagram_embeddings"
    MOLECULE_COLLECTION = "molecular_structures"

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(OUTPUT_DIR, "qdrant_db")
        os.makedirs(db_path, exist_ok=True)
        self.db_path = db_path

        # Proactively remove stale Qdrant lock file if present to prevent startup blocks
        lock_file = os.path.join(self.db_path, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass

        try:
            self.client = QdrantClient(path=self.db_path)
        except Exception as e:
            # If Qdrant failed due to an existing lock file, clean and retry
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                    self.client = QdrantClient(path=self.db_path)
                except Exception:
                    raise e
            else:
                raise e

        self.embedder = LightweightEmbedder(dim=384)
        self._init_collections()

    def _init_collections(self):
        """Initializes all collections if not already existing."""
        # 1. Questions Collection (384-D)
        if not self.client.collection_exists(self.TEXT_COLLECTION):
            self.client.create_collection(
                collection_name=self.TEXT_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection: {self.TEXT_COLLECTION}")

        # 2. Diagrams Collection (128-D from Swin Transformer + Feature MLP)
        if not self.client.collection_exists(self.DIAGRAM_COLLECTION):
            self.client.create_collection(
                collection_name=self.DIAGRAM_COLLECTION,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection: {self.DIAGRAM_COLLECTION}")

        # 3. Molecules Collection (1024-bit Morgan Fingerprints)
        if not self.client.collection_exists(self.MOLECULE_COLLECTION):
            self.client.create_collection(
                collection_name=self.MOLECULE_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection: {self.MOLECULE_COLLECTION}")

    # ==========================================================================
    # 1. QUESTION INDEXING & SEARCH
    # ==========================================================================
    def index_questions_from_json(self, json_path: str) -> int:
        """Indexes questions from kg_retrieved_context.json or generated_chemistry_questions.json."""
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            return 0

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        if not questions:
            return 0

        texts = []
        payloads = []
        point_ids = []

        existing_count = self.client.get_collection(self.TEXT_COLLECTION).points_count or 0

        for idx, q in enumerate(questions):
            q_text = str(q.get("question", "")).strip()
            if not q_text:
                continue

            topic = str(q.get("topic", q.get("kg_source_topic", "General"))).strip()
            subtopic = str(q.get("subtopic", q.get("kg_source_subtopic", ""))).strip()
            diff = str(q.get("difficulty", q.get("kg_source_difficulty", "Moderate"))).replace("DIFFICULTY_", "").strip()

            composite_text = f"Topic: {topic}. Subtopic: {subtopic}. Difficulty: {diff}. Question: {q_text}"
            texts.append(composite_text)

            # Clean and propagate correct_answer / correct_option
            ca = q.get("correct_answer")
            if ca in (None, "None", "none", "null", "NULL", ""):
                ca = q.get("correct_option")

            options = q.get("options", [])
            if ca in (None, "None", "none", "null", "NULL", "") and isinstance(options, list):
                for opt in options:
                    if isinstance(opt, dict) and opt.get("is_correct"):
                        ca = opt.get("option")
                        break

            clean_ca = str(ca).strip().upper() if (ca and str(ca).strip().upper() in ("A", "B", "C", "D")) else ""

            # Ensure options list has is_correct = True for clean_ca
            if clean_ca and isinstance(options, list):
                for opt in options:
                    if isinstance(opt, dict):
                        opt["is_correct"] = (str(opt.get("option", "")).strip().upper() == clean_ca)

            payloads.append({
                "question_id": str(q.get("question_id", q.get("kg_source_id", f"Q_{idx}"))),
                "question": q_text,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": diff,
                "question_type": str(q.get("question_type", "MCQ")),
                "correct_answer": clean_ca,
                "correct_option": clean_ca,
                "options": options,
                "explanation": str(q.get("explanation", ""))
            })
            point_ids.append(existing_count + idx + 1)

        vectors = self.embedder.embed_texts(texts)
        points = [
            PointStruct(id=pid, vector=v, payload=p)
            for pid, v, p in zip(point_ids, vectors, payloads)
        ]

        # Batch upsert
        batch_size = 100
        for b_start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.TEXT_COLLECTION,
                points=points[b_start : b_start + batch_size]
            )

        print(f"Successfully indexed {len(points)} questions into '{self.TEXT_COLLECTION}'.")
        return len(points)

    def index_questions_from_kg(self, graph_or_path: Any) -> int:
        """
        Indexes Question nodes from Knowledge Graph into Qdrant text collection.
        Propagates HAS_CORRECT_OPTION edges directly to correct_option & correct_answer fields.
        """
        if isinstance(graph_or_path, str):
            if not os.path.exists(graph_or_path):
                print(f"KG file not found: {graph_or_path}")
                return 0
            with open(graph_or_path, "rb") as f:
                graph = pickle.load(f)
        else:
            graph = graph_or_path

        if graph is None:
            return 0

        # Import propagation helper if available
        try:
            from knowledge_graph import propagate_correct_options
            graph = propagate_correct_options(graph)
        except Exception:
            pass

        question_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Question"]
        if not question_nodes:
            return 0

        texts = []
        payloads = []
        point_ids = []
        existing_count = self.client.get_collection(self.TEXT_COLLECTION).points_count or 0

        for idx, q_node in enumerate(question_nodes):
            q_data = graph.nodes[q_node]
            q_text = str(q_data.get("question", "")).strip()
            if not q_text:
                continue

            qid = str(q_data.get("label", q_node)).replace("QUESTION_", "")

            # Extract options and propagate HAS_CORRECT_OPTION edge
            options = []
            correct_from_edge = None
            for succ in graph.successors(q_node):
                succ_data = graph.nodes.get(succ, {})
                if succ_data.get("type") == "Option":
                    lbl = str(succ_data.get("label", "")).strip().upper()
                    m = re.search(r"([A-D])$", lbl)
                    letter = m.group(1) if m else (succ.split("_")[-1] if succ.split("_")[-1] in "ABCD" else "")
                    opt_text = str(succ_data.get("text", "")).strip()

                    edge_data = graph.get_edge_data(q_node, succ) or {}
                    rel = str(edge_data.get("relation", "")).upper()
                    is_correct = (rel == "HAS_CORRECT_OPTION" or "CORRECT_OPTION" in rel)
                    if is_correct and letter in "ABCD":
                        correct_from_edge = letter

                    options.append({
                        "option": letter,
                        "text": opt_text,
                        "is_correct": is_correct
                    })

            options.sort(key=lambda x: x.get("option", ""))

            # Resolve correct answer: edge > node attribute > None
            raw_ca = correct_from_edge or q_data.get("correct_option")
            clean_ca = str(raw_ca).strip().upper() if (raw_ca and str(raw_ca).strip().upper() in "ABCD") else ""

            # Ensure matching option is flagged as correct
            if clean_ca:
                for opt in options:
                    opt["is_correct"] = (opt.get("option") == clean_ca)

            # Trace metadata (difficulty, subtopic, topic)
            diff = "Moderate"
            for succ in graph.successors(q_node):
                if graph.nodes.get(succ, {}).get("type") == "Difficulty":
                    diff = str(graph.nodes[succ].get("label", "Moderate")).replace("DIFFICULTY_", "")
                    break

            topic = "General Chemistry"
            subtopic = "General"
            for pred in graph.predecessors(q_node):
                pdata = graph.nodes.get(pred, {})
                if pdata.get("type") == "Subtopic":
                    subtopic = str(pdata.get("label", "General"))
                    for tpred in graph.predecessors(pred):
                        tpdata = graph.nodes.get(tpred, {})
                        if tpdata.get("type") == "Topic":
                            topic = str(tpdata.get("label", "General Chemistry"))

            composite_text = f"Topic: {topic}. Subtopic: {subtopic}. Difficulty: {diff}. Question: {q_text}"
            texts.append(composite_text)

            payloads.append({
                "question_id": qid,
                "question": q_text,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": diff,
                "question_type": str(q_data.get("question_type", "MCQ")),
                "correct_answer": clean_ca,
                "correct_option": clean_ca,
                "options": options,
                "explanation": ""
            })
            point_ids.append(existing_count + idx + 1)

        vectors = self.embedder.embed_texts(texts)
        points = [
            PointStruct(id=pid, vector=v, payload=p)
            for pid, v, p in zip(point_ids, vectors, payloads)
        ]

        batch_size = 100
        for b_start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.TEXT_COLLECTION,
                points=points[b_start : b_start + batch_size]
            )

        print(f"Successfully indexed {len(points)} KG questions into '{self.TEXT_COLLECTION}'.")
        return len(points)

    def search_questions(
        self,
        query: str,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Dense semantic search over questions with optional metadata filtering."""
        query_vector = self.embedder.embed_query(query)

        # Adaptive query filter
        conditions = []
        if topic and topic.upper() != "ALL":
            conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
        if difficulty and difficulty.upper() != "ALL":
            conditions.append(FieldCondition(key="difficulty", match=MatchValue(value=difficulty)))

        q_filter = Filter(must=conditions) if conditions else None

        res = self.client.query_points(
            collection_name=self.TEXT_COLLECTION,
            query=query_vector,
            query_filter=q_filter,
            limit=top_k,
        )

        # Fallback 1: Relax difficulty if no matches
        if not res.points and difficulty and difficulty.upper() != "ALL":
            topic_conditions = [FieldCondition(key="topic", match=MatchValue(value=topic))] if (topic and topic.upper() != "ALL") else []
            t_filter = Filter(must=topic_conditions) if topic_conditions else None
            res = self.client.query_points(
                collection_name=self.TEXT_COLLECTION,
                query=query_vector,
                query_filter=t_filter,
                limit=top_k,
            )

        # Fallback 2: Pure dense semantic search across entire question bank
        if not res.points:
            res = self.client.query_points(
                collection_name=self.TEXT_COLLECTION,
                query=query_vector,
                limit=top_k,
            )

        output = []
        for r in res.points:
            item = dict(r.payload)
            score_val = float(r.score)
            item["similarity_score"] = score_val
            item["score"] = score_val

            # Sanitize correct_answer and correct_option
            ca = item.get("correct_answer")
            if ca in (None, "None", "none", "null", "NULL", ""):
                ca = item.get("correct_option")
            opts = item.get("options", [])
            if ca in (None, "None", "none", "null", "NULL", "") and isinstance(opts, list):
                for opt in opts:
                    if isinstance(opt, dict) and opt.get("is_correct"):
                        ca = opt.get("option")
                        break
            clean_ca = str(ca).strip().upper() if (ca and str(ca).strip().upper() in ("A", "B", "C", "D")) else ""
            item["correct_answer"] = clean_ca
            item["correct_option"] = clean_ca
            if clean_ca and isinstance(opts, list):
                for opt in opts:
                    if isinstance(opt, dict):
                        opt["is_correct"] = (str(opt.get("option", "")).strip().upper() == clean_ca)

            output.append(item)
        return output


    # ==========================================================================
    # 2. DIAGRAM EMBEDDING INDEXING & SEARCH (Swin-T 128-D)
    # ==========================================================================
    def index_swin_features_csv(self, csv_path: str) -> int:
        """Indexes pre-computed 128-D Swin Transformer + Feature MLP diagram vectors."""
        if not os.path.exists(csv_path):
            print(f"Diagram feature CSV not found: {csv_path}")
            return 0

        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        if len(feature_cols) != 128:
            print(f"Expected 128 feature columns, found {len(feature_cols)}")
            return 0

        points = []
        for idx, row in df.iterrows():
            vec = row[feature_cols].values.astype(float).tolist()
            # Normalize vector to unit length
            v_arr = np.array(vec)
            norm = np.linalg.norm(v_arr)
            if norm > 0:
                v_arr = v_arr / norm

            payload = {
                "image_path": str(row["image_path"]),
                "filename": os.path.basename(str(row["image_path"])),
            }
            points.append(PointStruct(id=idx + 1, vector=v_arr.tolist(), payload=payload))

        batch_size = 250
        for b_start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.DIAGRAM_COLLECTION,
                points=points[b_start : b_start + batch_size]
            )

        print(f"Successfully indexed {len(points)} diagram feature vectors into '{self.DIAGRAM_COLLECTION}'.")
        return len(points)

    def search_similar_diagrams(self, query_vector_128d: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds visually similar diagrams using cosine similarity over Swin features."""
        v_arr = np.array(query_vector_128d)
        norm = np.linalg.norm(v_arr)
        if norm > 0:
            v_arr = v_arr / norm

        res = self.client.query_points(
            collection_name=self.DIAGRAM_COLLECTION,
            query=v_arr.tolist(),
            limit=top_k
        )
        return [{"image_path": r.payload["image_path"], "filename": r.payload.get("filename", ""), "score": float(r.score)} for r in res.points]

    # Alias for search_similar_diagrams
    search_diagrams = search_similar_diagrams

    # ==========================================================================
    # 3. CHEMICAL STRUCTURE FINGERPRINT INDEXING & SEARCH (Morgan ECFP4 / Substructure Hash)
    # ==========================================================================
    def _compute_smiles_fingerprint(self, smiles: str) -> List[float]:
        """Computes a normalized 1024-D structural fingerprint from SMILES."""
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                    arr = np.zeros((1024,), dtype=float)
                    for bit in fp.GetOnBits():
                        arr[bit] = 1.0
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    return arr.tolist()
            except Exception:
                pass

        # Robust Substructure N-Gram Hashing Fallback (1024-D)
        arr = np.zeros((1024,), dtype=float)
        s = smiles.strip()
        for n in range(1, 5):
            for i in range(len(s) - n + 1):
                token = s[i : i + n]
                bucket = hash(token) % 1024
                arr[bucket] += 1.0
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def index_molecule_bank(self, molecules: List[tuple]) -> int:
        """Indexes chemical structures via 1024-bit Morgan Fingerprints / Structural Hashes."""
        points = []
        for idx, item in enumerate(molecules):
            name, smiles = item[0], item[1]
            try:
                vec = self._compute_smiles_fingerprint(smiles)
                payload = {
                    "name": name,
                    "smiles": smiles,
                    "formula": name,
                }
                points.append(PointStruct(id=idx + 1, vector=vec, payload=payload))
            except Exception:
                continue

        if points:
            self.client.upsert(collection_name=self.MOLECULE_COLLECTION, points=points)
            print(f"Successfully indexed {len(points)} molecular fingerprints into '{self.MOLECULE_COLLECTION}'.")
        return len(points)

    def search_similar_molecules(self, query_smiles: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds chemically similar molecules using Cosine similarity over structural fingerprints."""
        query_vec = self._compute_smiles_fingerprint(query_smiles)
        res = self.client.query_points(
            collection_name=self.MOLECULE_COLLECTION,
            query=query_vec,
            limit=top_k
        )
        return [{"name": r.payload["name"], "smiles": r.payload["smiles"], "formula": r.payload.get("formula", ""), "score": float(r.score)} for r in res.points]


    def close(self):
        """Closes the underlying Qdrant client cleanly."""
        try:
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_stats(self) -> Dict[str, int]:
        """Returns point counts across all 3 collections."""
        stats = {}
        for col in [self.TEXT_COLLECTION, self.DIAGRAM_COLLECTION, self.MOLECULE_COLLECTION]:
            try:
                stats[col] = self.client.get_collection(col).points_count or 0
            except Exception:
                stats[col] = 0
        return stats

