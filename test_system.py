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
import json
import unittest

from config import (
    AGENT4_OUTPUT_JSON,
    AGENT4_OUTPUT_HTML,
    AGENT4_OUTPUT_DOCX,
)
from agent_1_requirement_analysis import RequirementAnalysisAgent
from vector_store import ChemistryVectorStore
from hybrid_graph_rag import HybridGraphRAGEngine
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
        for q in context["questions"]:
            self.assertNotEqual(q.get("correct_answer"), "None")
            self.assertNotEqual(q.get("correct_option"), "None")
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
        # Assertion-Reason format (verifies diverse keys A, B, C, D via Distractor Agent)
        qs_ar = agent4.generate(total_count=4, mode="image", topic="Chemical Kinetics", question_format="Assertion-Reason")
        self.assertEqual(len(qs_ar), 4)
        for q in qs_ar:
            self.assertIn("Assertion (A)", q["question"])
            self.assertIn("Reason (R)", q["question"])
            self.assertIn(q["correct_answer"], ["A", "B", "C", "D"])
        self.assertEqual([q["correct_answer"] for q in qs_ar], ["A", "B", "C", "D"])

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

    def test_10_kg_has_correct_option_propagation(self):
        """Test HAS_CORRECT_OPTION edges propagate to correct_option field in KG and Vector Store."""
        import networkx as nx
        from knowledge_graph import propagate_correct_options

        # 1. Build synthetic test graph with edge relation HAS_CORRECT_OPTION and None correct_option
        G = nx.DiGraph()
        G.add_node("QUESTION_TEST_Q1", type="Question", label="TEST_Q1", question="Sample rate law question", correct_option=None)
        G.add_node("OPTION_TEST_Q1_A", type="Option", label="Option A", text="Option A text")
        G.add_node("OPTION_TEST_Q1_B", type="Option", label="Option B", text="Option B text")
        G.add_edge("QUESTION_TEST_Q1", "OPTION_TEST_Q1_A", relation="HAS_OPTION")
        G.add_edge("QUESTION_TEST_Q1", "OPTION_TEST_Q1_B", relation="HAS_CORRECT_OPTION")

        # Verify propagation from edge to Question node
        G = propagate_correct_options(G)
        self.assertEqual(G.nodes["QUESTION_TEST_Q1"].get("correct_option"), "B")

        # 2. Test propagation through Vector Store indexing
        vs = ChemistryVectorStore()
        indexed = vs.index_questions_from_kg(G)
        self.assertGreaterEqual(indexed, 1)

        # 3. Verify indexed point has correct_answer == "B" and option B is_correct == True
        results = vs.search_questions("Sample rate law question", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].get("correct_answer"), "B")
        self.assertEqual(results[0].get("correct_option"), "B")
        b_opts = [o for o in results[0].get("options", []) if o.get("option") == "B"]
        self.assertTrue(len(b_opts) > 0 and b_opts[0].get("is_correct"))
        vs.close()
        print("\n[PASS] HAS_CORRECT_OPTION edge propagation to correct_option field verified.")

    def test_11_deterministic_validators(self):
        """Test DeterministicValidator recomputes answers programmatically without LLM."""
        from verification.deterministic_validator import DeterministicValidator

        # 1. First-order kinetics (rate constant)
        q_fo = {
            "image_family": "g_first_order",
            "question": "The graph shows a first-order reaction ln([A]/[A]₀) vs time. Point (60, -1.20).",
            "option_A": "0.0200 min⁻¹",
            "option_B": "0.0400 min⁻¹",
            "correct_answer": "A",
            "explanation": "Slope = -k. Here slope = -1.20/60 = -0.0200, so k = 0.0200 min⁻¹."
        }
        passed, msg = DeterministicValidator.validate(q_fo)
        self.assertTrue(passed, f"First-order validation failed: {msg}")

        # 2. Arrhenius activation energy
        q_arr = {
            "image_family": "g_arrhenius",
            "question": "Arrhenius plot (ln k vs 1/T) with slope. Coordinates (0.002, 10.0) and (0.003, -1.5).",
            "option_A": "95 kJ mol⁻¹",
            "correct_answer": "A",
            "explanation": "Slope = -Ea/R = -11500. Ea = 95 kJ/mol."
        }
        passed, msg = DeterministicValidator.validate(q_arr)
        self.assertTrue(passed, f"Arrhenius validation failed: {msg}")

        # 3. Stoichiometry / combustion
        q_comb = {
            "image_family": "org_combustion",
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin C9H8O4 + 9 O2 -> 9 CO2 + 4 H2O
            "question": "If 0.50 mol undergoes complete combustion, the amount of O₂ required is",
            "option_A": "4.5 mol",
            "correct_answer": "A",
            "explanation": "O₂ per mole = 9 + 8/4 - 4/2 = 9. Total = 0.50 × 9 = 4.5 mol."
        }
        passed, msg = DeterministicValidator.validate(q_comb)
        self.assertTrue(passed, f"Stoichiometry validation failed: {msg}")

        # 4. VSEPR geometry
        q_vsepr = {
            "image_family": "inorg_shape",
            "smiles": "FS(F)(F)F",  # SF4 -> See-saw
            "question": "The molecular geometry of SF4 is",
            "option_A": "See-saw",
            "correct_answer": "A",
            "explanation": "SF4 has 4 bond pairs and 1 lone pair -> See-saw shape."
        }
        passed, msg = DeterministicValidator.validate(q_vsepr)
        self.assertTrue(passed, f"VSEPR validation failed: {msg}")

        # 5. Numerical format validator rejects magnitude < 1
        q_bad_num = {
            "question_format": "Numerical",
            "correct_answer": "0",
            "question": "The rate constant is [Answer in nearest integer]"
        }
        passed, msg = DeterministicValidator.validate(q_bad_num)
        self.assertFalse(passed)
        self.assertIn("magnitude < 1", msg)

        print("\n[PASS] DeterministicValidator programmatic recomputation verified.")

    def test_12_specialized_agents_and_cross_chapter_pairer(self):
        """Test Constraint, Formulation, Distractor, Verification agents and KG Cross-Chapter Pairer."""
        from specialized_agents import ConstraintAgent, FormulationAgent, DistractorAgent, VerificationAgent
        from cross_chapter_pairer import CrossChapterPairer

        # 1. Constraint Agent
        c_agent = ConstraintAgent()
        params = c_agent.choose_parameters("g_first_order", difficulty="Hard")
        self.assertTrue(c_agent.is_realistic(params))
        self.assertGreater(params["rate_constant"], 0)

        # 2. Distractor Agent
        d_agent = DistractorAgent()
        dist_res = d_agent.generate("g_first_order", "0.0200 min⁻¹", {"rate_constant": 0.02})
        self.assertEqual(len(dist_res["distractors"]), 3)
        self.assertIn("diagnostics", dist_res)

        # 3. Formulation Agent
        f_agent = FormulationAgent()
        stem = f_agent.write_stem(params, "g_first_order", question_format="MCQ")
        self.assertIn("first-order", stem.lower())
        opts = f_agent.format_options(dist_res["correct"], dist_res["distractors"], target_key="B")
        self.assertEqual(opts["correct_answer"], "B")
        self.assertEqual(opts["option_B"], dist_res["correct"])

        # 4. Verification Agent Revision
        v_agent = VerificationAgent()
        bad_q = {"question_format": "Numerical", "correct_answer": "0", "question": "Rate constant", "explanation": "k = 0.025 min⁻¹"}
        revised = v_agent.revise(bad_q, "Numerical answer key 0 has magnitude < 1.")
        self.assertGreaterEqual(int(revised["correct_answer"]), 1)

        # 5. Cross-Chapter Pairer via KG Traversal
        pairer = CrossChapterPairer()
        pairs = pairer.find_pairs("Chemical Kinetics", max_pairs=2)
        self.assertGreaterEqual(len(pairs), 1)
        self.assertIn("coupling_formula", pairs[0])
        self.assertIn("description", pairs[0])

        print("\n[PASS] Specialized Agents and Cross-Chapter KG Pairer verified.")


if __name__ == "__main__":
    unittest.main()
