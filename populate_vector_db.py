"""
populate_vector_db.py
Populates the Qdrant Tri-Modal Vector Database with:
  1. JEE chemistry questions (text embeddings via FastEmbed BGE-small)
  2. Diagram visual feature vectors (128-D Swin-T embeddings)
  3. Chemical structures (1024-bit RDKit Morgan fingerprints)
"""

import os
import sys
from vector_store import ChemistryVectorStore
from agent_4_question_generation import ORGANIC_BANK, INORGANIC_BANK


def main():
    print("=" * 70)
    print("INITIALIZING AND POPULATING TRI-MODAL QDRANT VECTOR DATABASE")
    print("=" * 70)

    vs = ChemistryVectorStore()

    # 1. Index Questions
    print("\n[1/3] Indexing Question Banks...")
    q1 = vs.index_questions_from_json("outputs/kg_retrieved_context.json")
    q2 = vs.index_questions_from_json("outputs/generated_chemistry_questions.json")
    print(f"Total questions indexed: {q1 + q2}")

    # 2. Index Swin Transformer Diagram Features
    print("\n[2/3] Indexing Diagram Visual Feature Vectors...")
    n_diag = vs.index_swin_features_csv("outputs/chemistry_swin_mlp_features.csv")
    print(f"Total diagram vectors indexed: {n_diag}")

    # 3. Index Molecular Structures
    print("\n[3/3] Indexing Chemical Structures (Morgan Fingerprints)...")
    n_mols = vs.index_molecule_bank(ORGANIC_BANK + INORGANIC_BANK)
    print(f"Total molecules indexed: {n_mols}")

    print("\n" + "=" * 70)
    print("DATABASE STATUS & BENCHMARK TESTS")
    print("=" * 70)
    stats = vs.get_stats()
    print(f"Qdrant Collections: {stats}")

    # Benchmark Test 1: Semantic Question Search
    print("\n>>> Test 1: Semantic Search for 'Arrhenius activation energy rate constant':")
    res = vs.search_questions("Arrhenius activation energy rate constant", top_k=3)
    for i, r in enumerate(res, 1):
        print(f"  {i}. [{r.get('question_id')}] (Score: {r.get('similarity_score', 0):.4f}) {r.get('question', '')[:90]}...")

    # Benchmark Test 2: Chemical Molecule Similarity Search
    print("\n>>> Test 2: Chemical Structure Search for Aspirin (SMILES: CC(=O)Oc1ccccc1C(=O)O):")
    mols = vs.search_similar_molecules("CC(=O)Oc1ccccc1C(=O)O", top_k=3)
    for i, m in enumerate(mols, 1):
        print(f"  {i}. {m['name']} (Score: {m['score']:.4f}) - Formula: {m['formula']}")

    # Benchmark Test 3: Filtered Search by Topic & Difficulty
    print("\n>>> Test 3: Filtered Semantic Search for 'Equilibrium', Difficulty='Moderate':")
    filt_res = vs.search_questions("equilibrium constant partial pressure", topic="Physical Chemistry", difficulty="Moderate", top_k=2)
    for i, r in enumerate(filt_res, 1):
        print(f"  {i}. [{r.get('question_id')}] Topic: {r.get('topic')} | {r.get('question', '')[:80]}...")

    vs.close()
    print("\n" + "=" * 70)
    print("TRI-MODAL VECTOR STORE POPULATION & VERIFICATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
