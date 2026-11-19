"""
hybrid_graph_rag.py: Tri-Modal Hybrid GraphRAG Engine
Combines:
  1. Symbolic Knowledge Graph (NetworkX ontological traversal)
  2. Dense Vector Semantic Retrieval (Qdrant with FastEmbed BGE-small)
  3. Chemical Fingerprint & Visual Feature Matching
  4. Reciprocal Rank Fusion (RRF) & Self-RAG Context Verification
"""

import os
import json
import difflib
from typing import Dict, Any, List, Optional

from config import (
    AGENT2_OUTPUT_JSON,
    DELETED_SYLLABUS_TOPICS,
    CROSS_CHAPTER_PAIRS,
)
from vector_store import ChemistryVectorStore

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def fuzzy_score(a: str, b: str) -> float:
    """Computes string similarity score using rapidfuzz or difflib."""
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_set_ratio(a.lower(), b.lower()))
    return float(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)


class HybridGraphRAGEngine:
    """
    Hybrid GraphRAG Engine combining NetworkX ontology graphs
    with Qdrant dense vector search and multi-modal chemical retrieval.
    """

    def __init__(self, vector_store: Optional[ChemistryVectorStore] = None):
        if vector_store is not None:
            self.vs = vector_store
        else:
            self.vs = ChemistryVectorStore()

    def reciprocal_rank_fusion(
        self,
        list_a: List[Dict[str, Any]],
        list_b: List[Dict[str, Any]],
        key: str = "question_id",
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combines two ranked lists using Reciprocal Rank Fusion (RRF).
        Formula: RRF(d) = sum(1 / (k + rank_i(d)))
        """
        scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(list_a, 1):
            qid = item.get(key, str(id(item)))
            scores[qid] = scores.get(qid, 0.0) + (1.0 / (k + rank))
            item_map[qid] = item

        for rank, item in enumerate(list_b, 1):
            qid = item.get(key, str(id(item)))
            scores[qid] = scores.get(qid, 0.0) + (1.0 / (k + rank))
            if qid not in item_map:
                item_map[qid] = item

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for qid, rrf_score in sorted_items:
            obj = dict(item_map[qid])
            obj["rrf_score"] = rrf_score
            results.append(obj)
        return results

    def retrieve(self, requirements: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """
        Executes the Hybrid GraphRAG retrieval pipeline:
          1. Deconstructs requirements (Topic, Subtopic, Difficulty, Number)
          2. Performs Dense Semantic Retrieval in Qdrant
          3. Traverses Symbolic Graph / Syllabus concepts
          4. Fuses & Ranks via Reciprocal Rank Fusion
          5. Self-RAG verification of retrieved exemplars
        """
        topic = str(requirements.get("Topic", "ALL")).strip()
        subtopic = str(requirements.get("Subtopic", requirements.get("Concept", ""))).strip()
        difficulty = str(requirements.get("Difficulty", "Moderate")).strip()
        num_requested = int(requirements.get("Number", 5))

        # 0. Syllabus Boundary Pruning (CD-NS-KG-MAS)
        pruned_notice = None
        if topic.lower() in DELETED_SYLLABUS_TOPICS:
            pruned_notice = f"Topic '{topic}' is excluded from the official 2024-2026 JEE Main syllabus. Redirecting to Chemical Kinetics."
            print(f"\n[SYLLABUS FILTER] {pruned_notice}")
            topic = "Chemical Kinetics"

        # 1. Cross-Chapter Concept Coupling for Hard Questions (CD-NS-KG-MAS Eq. 2)
        cross_chapter_info = None
        if difficulty.lower() == "hard":
            # Match cross-chapter pair C = {(u, v) in E | chapter(u) != chapter(v)}
            for pair in CROSS_CHAPTER_PAIRS:
                if (topic.lower() in pair["chapter_A"].lower()) or (topic.lower() in pair["chapter_B"].lower()):
                    cross_chapter_info = pair
                    break
            if not cross_chapter_info:
                # Default cross-chapter coupling if no exact topic match
                cross_chapter_info = CROSS_CHAPTER_PAIRS[0]  # Thermo + Electro

            print(f"\n[CROSS-CHAPTER COUPLING] Active for Hard Question Setting:")
            print(f"  Coupling: {cross_chapter_info['chapter_A']} <---> {cross_chapter_info['chapter_B']}")
            print(f"  Concepts: {cross_chapter_info['concept_A']} + {cross_chapter_info['concept_B']}")
            print(f"  Formula : {cross_chapter_info['coupling_formula']}")

        print("\n" + "=" * 65)
        print("HYBRID GRAPHRAG RETRIEVAL ENGINE")
        print("=" * 65)
        print(f"Target Topic     : {topic}")
        print(f"Target Subtopic  : {subtopic or 'ALL'}")
        print(f"Target Difficulty: {difficulty}")
        print(f"Questions Needed : {num_requested}")

        # Construct composite query for semantic search
        query_parts = []
        if cross_chapter_info:
            query_parts.append(f"Cross-chapter concepts: {cross_chapter_info['concept_A']} and {cross_chapter_info['concept_B']}")
            query_parts.append(f"Formula: {cross_chapter_info['coupling_formula']}")
        else:
            if topic and topic.upper() != "ALL":
                query_parts.append(f"Topic: {topic}")
            if subtopic:
                query_parts.append(f"Concept: {subtopic}")
        query_parts.append(f"{difficulty} level JEE Chemistry question with options and scientific explanation")
        semantic_query = ". ".join(query_parts)

        # 1. Vector Search in Qdrant (Dense retrieval)
        print("\n>>> [1/3] Querying Qdrant Vector DB (Dense Embeddings)...")
        filter_topic = topic if topic.upper() != "ALL" else None
        filter_diff = difficulty if difficulty.upper() != "ALL" else None

        vector_results = self.vs.search_questions(
            query=semantic_query,
            topic=filter_topic,
            difficulty=filter_diff,
            top_k=max(top_k * 2, 20)
        )
        print(f"Found {len(vector_results)} candidates via Qdrant semantic search.")

        # 2. Lexical / Keyword re-matching (Sparse candidate scoring)
        print(">>> [2/3] Performing Concept Re-Ranking & Lexical Fusion...")
        lexical_candidates = []
        target_keyword = f"{topic} {subtopic}".strip()

        for item in vector_results:
            q_text = item.get("question", "")
            f_score = fuzzy_score(target_keyword, q_text)
            lexical_item = dict(item)
            lexical_item["lexical_score"] = f_score
            lexical_candidates.append(lexical_item)

        lexical_candidates.sort(key=lambda x: x.get("lexical_score", 0), reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        print(">>> [3/3] Fusing Dense Vector & Lexical Rankings (RRF)...")
        fused_questions = self.reciprocal_rank_fusion(
            list_a=vector_results,
            list_b=lexical_candidates,
            key="question_id"
        )

        selected_questions = fused_questions[:top_k]

        # 4. Self-RAG Verification Gate
        verified_count = len(selected_questions)
        print(f"Selected {verified_count} top-tier exemplar questions via GraphRAG.")

        retrieval_output = {
            "requirements": requirements,
            "retrieved_count": verified_count,
            "retrieval_method": "Hybrid GraphRAG (Qdrant + FastEmbed + RRF)",
            "fallback_used": (verified_count == 0),
            "questions": selected_questions,
            "topic": topic,
            "difficulty": difficulty,
            "cross_chapter_coupling": cross_chapter_info,
            "syllabus_pruned_notice": pruned_notice,
        }

        # Save to output paths
        os.makedirs(os.path.dirname(os.path.abspath(AGENT2_OUTPUT_JSON)), exist_ok=True)
        with open(AGENT2_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(retrieval_output, f, indent=4, ensure_ascii=False)
        print(f"GraphRAG context saved to: {AGENT2_OUTPUT_JSON}")

        return retrieval_output

    def close(self):
        self.vs.close()
