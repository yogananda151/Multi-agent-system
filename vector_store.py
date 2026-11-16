"""
vector_store.py: Tri-Modal Hybrid Vector Store for JEE Chemistry System
Manages Qdrant collections for:
  1. chemistry_questions: Dense semantic vectors + payload filtering
  2. diagram_embeddings: 128-D Swin Transformer + Feature MLP vectors
  3. molecular_structures: 1024-bit Morgan Chemical Fingerprints (ECFP4)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from config import OUTPUT_DIR, CLEANED_DATA_PATH

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
        self.client = QdrantClient(path=self.db_path)
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

            payloads.append({
                "question_id": str(q.get("question_id", q.get("kg_source_id", f"Q_{idx}"))),
                "question": q_text,
                "topic": topic,
                "subtopic": subtopic,
                "difficulty": diff,
                "question_type": str(q.get("question_type", "MCQ")),
                "correct_answer": str(q.get("correct_answer", q.get("correct_option", ""))),
                "options": q.get("options", []),
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
            item["similarity_score"] = float(r.score)
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
        return [{"image_path": r.payload["image_path"], "score": float(r.score)} for r in res.points]

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

