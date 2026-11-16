"""
test_system.py: Comprehensive System Verification Suite
Validates:
  1. Agent 1: Natural Language Requirement Analysis
  2. Tri-Modal Qdrant Vector Store (dense vectors, diagram embeddings, molecules)
  3. Agent 2: Hybrid GraphRAG Retrieval Engine & RRF
  4. Agent 3: Examination Prompt Construction
  5. Agent 4: Multimodal Chemistry Question Generation (Matplotlib & RDKit)
  6. Pipeline: End-to-End Orchestration & Multi-Format Exporters
"""

import os
import sys
import json
import unittest

from config import (
    OUTPUT_DIR,
    AGENT1_REQUIREMENTS_PATH,
    AGENT2_OUTPUT_JSON,
    AGENT3_OUTPUT_TXT,
    AGENT4_OUTPUT_JSON,
    AGENT4_OUTPUT_HTML,
    AGENT4_OUTPUT_DOCX,
)
from agent_1_requirement_analysis import RequirementAnalysisAgent
from vector_store import ChemistryVectorStore
from hybrid_graph_rag import HybridGraphRAGEngine
from agent_2_kg_retrieval import KnowledgeGraphRetrievalAgent
from agent_3_prompt_construction import PromptConstructionAgent
from agent_4_question_generation import MultimodalChemistryAgent
from pipeline import ChemistryMultiAgentPipeline


class TestChemistrySystem(unittest.TestCase):

    def test_01_agent_1_requirement_parsing(self):
        """Test Agent 1 natural language query parsing."""
        agent = RequirementAnalysisAgent()
        reqs = agent.analyze("Generate 7 hard MCQs on Chemical Kinetics")
        self.assertEqual(reqs.get("Number"), 7)
        self.assertEqual(reqs.get("Difficulty"), "Hard")
        self.assertIn("Chemical Kinetics", reqs.get("Topic"))
        print("\n[PASS] Agent 1 requirement analysis passed.")

    def test_02_vector_store_collections(self):
        """Test Qdrant vector store collections and search."""
        vs = ChemistryVectorStore()
        stats = vs.get_stats()
        self.assertGreater(stats.get("chemistry_questions", 0), 0)
        self.assertGreater(stats.get("diagram_embeddings", 0), 0)
        self.assertGreater(stats.get("molecular_structures", 0), 0)

        # Test query
        results = vs.search_questions("rate law kinetics", top_k=2)
        self.assertGreater(len(results), 0)
        vs.close()
        print(f"\n[PASS] Vector store collections verified. Stats: {stats}")

    def test_03_hybrid_graph_rag_engine(self):
        """Test Hybrid GraphRAG retrieval with RRF ranking."""
        rag = HybridGraphRAGEngine()
        reqs = {"Topic": "Chemical Kinetics", "Difficulty": "Moderate", "Number": 3}
        context = rag.retrieve(reqs, top_k=3)
        self.assertIn("questions", context)
        self.assertGreaterEqual(len(context["questions"]), 1)
        self.assertEqual(context.get("retrieval_method"), "Hybrid GraphRAG (Qdrant + FastEmbed + RRF)")
        rag.close()
        print("\n[PASS] Hybrid GraphRAG retrieval & RRF passed.")

    def test_04_agent_3_prompt_construction(self):
        """Test Agent 3 assembling prompt with few-shot exemplars."""
        agent3 = PromptConstructionAgent()
        reqs = {"Number": 3, "Topic": "Chemical Kinetics", "Difficulty": "Moderate", "Type": "MCQ"}
        context = {"questions": [{"question_id": "TEST_01", "question": "What is reaction order?", "difficulty": "Moderate"}]}
        prompt = agent3.build_prompt(reqs, context)
        self.assertIn("TARGET REQUIREMENTS", prompt)
        self.assertIn("Chemical Kinetics", prompt)
        print("\n[PASS] Agent 3 prompt construction passed.")

    def test_05_agent_4_multimodal_generation(self):
        """Test Agent 4 multimodal image question generation & exports."""
        agent4 = MultimodalChemistryAgent()
        questions = agent4.generate(total_count=2, mode="image", topic="Chemical Kinetics")
        self.assertEqual(len(questions), 2)
        self.assertTrue(os.path.exists(AGENT4_OUTPUT_JSON))
        self.assertTrue(os.path.exists(AGENT4_OUTPUT_HTML))
        self.assertTrue(os.path.exists(AGENT4_OUTPUT_DOCX))
        print("\n[PASS] Agent 4 multimodal question generation & export passed.")

    def test_06_end_to_end_pipeline(self):
        """Test master pipeline orchestration."""
        pipe = ChemistryMultiAgentPipeline()
        qs = pipe.run(user_request="Generate 2 moderate MCQs on Chemical Kinetics", mode="image")
        self.assertEqual(len(qs), 2)
        print("\n[PASS] End-to-end Master Pipeline passed.")

    def test_07_cd_ns_kg_mas_cross_chapter_and_diagnostics(self):
        """Test CD-NS-KG-MAS cross-chapter coupling, distractor diagnostics, and SFT dataset export."""
        rag = HybridGraphRAGEngine()
        reqs = {"Topic": "Chemical Kinetics", "Difficulty": "Hard", "Number": 2}
        context = rag.retrieve(reqs, top_k=2)
        self.assertIsNotNone(context.get("cross_chapter_coupling"))
        self.assertEqual(context["cross_chapter_coupling"]["pair_id"], "X_KINETICS_EQUILIBRIUM")
        rag.close()

        # Test generation with diagnostics and SFT export
        agent4 = MultimodalChemistryAgent()
        qs = agent4.generate(total_count=2, mode="image", topic="Chemical Kinetics", difficulty="Hard")
        self.assertEqual(len(qs), 2)
        self.assertIn("distractor_diagnostics", qs[0])
        self.assertGreater(len(qs[0]["distractor_diagnostics"]), 0)

        # Verify SFT JSONL dataset
        from config import SFT_OUTPUT_JSONL
        self.assertTrue(os.path.exists(SFT_OUTPUT_JSONL))
        with open(SFT_OUTPUT_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreaterEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertIn("instruction", record)
        self.assertIn("input", record)
        self.assertIn("output", record)
        print("\n[PASS] CD-NS-KG-MAS Cross-Chapter coupling, Distractor Diagnostics & SFT JSONL passed.")

    def test_08_multi_format_generation(self):
        """Test Assertion-Reason and Numerical Integer question generation."""
        agent4 = MultimodalChemistryAgent()
        # Assertion-Reason format
        qs_ar = agent4.generate(total_count=1, mode="image", topic="Chemical Kinetics", question_format="Assertion-Reason")
        self.assertEqual(len(qs_ar), 1)
        self.assertIn("Assertion (A)", qs_ar[0]["question"])
        self.assertIn("Reason (R)", qs_ar[0]["question"])
        self.assertEqual(qs_ar[0]["correct_answer"], "A")

        # Numerical Integer format
        qs_num = agent4.generate(total_count=1, mode="image", topic="Chemical Kinetics", question_format="Numerical")
        self.assertEqual(len(qs_num), 1)
        self.assertIn("integer", qs_num[0]["question"].lower())
        self.assertTrue(qs_num[0]["correct_answer"].isdigit() or qs_num[0]["correct_answer"].lstrip("-").isdigit())
        print("\n[PASS] Multi-format (Assertion-Reason & Numerical) question generation passed.")

    def test_09_fullstack_computer_vision_suite(self):
        """Test Fullstack Computer Vision models (U-Net, Swin-T, Classifier, Reverse CBIR)."""
        from PIL import Image
        import vision_engine

        dummy = Image.new("L", (160, 128), color=235)

        # 1. U-Net Denoising
        restorer = vision_engine.DiagramRestorationEngine()
        restored, metrics = restorer.denoise_and_restore(dummy)
        self.assertEqual(restored.size, (160, 128))
        self.assertIn("contrast_improvement_pct", metrics)

        # 2. Swin-T Feature Extractor
        extractor = vision_engine.SwinDiagramFeatureExtractor()
        feat = extractor.extract_features(dummy)
        self.assertEqual(len(feat), 128)

        # 3. Diagram Classifier
        classifier = vision_engine.DiagramClassifier()
        c_res = classifier.classify(dummy)
        self.assertIn("predicted_family", c_res)
        self.assertGreater(c_res["confidence"], 0.70)

        # 4. Reverse Diagram Search
        searcher = vision_engine.ReverseDiagramSearchEngine()
        matches = searcher.search_by_image(dummy, top_k=2)
        self.assertGreaterEqual(len(matches), 1)
        searcher.close()
        print("\n[PASS] Fullstack Computer Vision Suite (U-Net, Swin-T, Classifier, CBIR) passed.")


if __name__ == "__main__":
    unittest.main()
