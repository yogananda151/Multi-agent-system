"""
End-to-End Orchestration Pipeline for Chemistry Multi-Agent System
Connects Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 in an automated workflow.
"""

import os
import json
import argparse

from config import (
    OUTPUT_DIR,
    KG_PICKLE_PATH,
    AGENT1_REQUIREMENTS_PATH,
    AGENT2_OUTPUT_JSON,
    AGENT3_OUTPUT_TXT,
    AGENT4_OUTPUT_JSON,
    AGENT4_OUTPUT_HTML,
    AGENT4_OUTPUT_DOCX,
)
from agent_1_requirement_analysis import RequirementAnalysisAgent, load_dataset_records
from agent_2_kg_retrieval import KnowledgeGraphRetrievalAgent
from agent_3_prompt_construction import PromptConstructionAgent
from agent_4_question_generation import MultimodalChemistryAgent
from knowledge_graph import ChemistryKnowledgeGraph


class ChemistryMultiAgentPipeline:
    """Full pipeline coordinator for JEE Chemistry Question Generation."""

    def __init__(self):
        self.output_dir = OUTPUT_DIR

    def run(
        self,
        user_request: str = None,
        mode: str = "mixed",
        total_count: int = None,
        image_count: int = None,
        difficulty: str = None,
        topic: str = None,
    ):
        print("\n" + "=" * 70)
        print("CHEMISTRY MULTI-AGENT QUESTION GENERATION PIPELINE")
        print("=" * 70)

        # ----------------------------------------------------------------------
        # STEP 1: REQUIREMENT ANALYSIS (AGENT 1)
        # ----------------------------------------------------------------------
        print("\n>>> [STEP 1/4] Running Agent 1: Requirement Analysis...")
        records = None
        try:
            records = load_dataset_records()
        except Exception:
            pass

        agent_1 = RequirementAnalysisAgent(records)
        if not user_request:
            user_request = "Generate 5 moderate MCQs across syllabus"

        requirements = agent_1.analyze(user_request)

        # Apply CLI overrides if user explicitly specified them
        if total_count is not None:
            requirements["Number"] = total_count
        if difficulty is not None:
            requirements["Difficulty"] = difficulty
        if topic is not None:
            requirements["Topic"] = topic

        agent_1.save_requirements(requirements, AGENT1_REQUIREMENTS_PATH)
        print(f"Agent 1 Requirements:\n{json.dumps(requirements, indent=2)}")

        # ----------------------------------------------------------------------
        # STEP 2: KNOWLEDGE GRAPH RETRIEVAL (AGENT 2)
        # ----------------------------------------------------------------------
        print("\n>>> [STEP 2/4] Running Agent 2: Knowledge Graph Retrieval...")
        agent_2 = KnowledgeGraphRetrievalAgent()
        retrieved_context = agent_2.retrieve(requirements)
        print(f"Agent 2 retrieved {retrieved_context.get('retrieved_count', 0)} reference questions.")

        # ----------------------------------------------------------------------
        # STEP 3: PROMPT CONSTRUCTION (AGENT 3)
        # ----------------------------------------------------------------------
        print("\n>>> [STEP 3/4] Running Agent 3: Prompt Construction...")
        agent_3 = PromptConstructionAgent()
        prompt = agent_3.build_prompt(requirements, retrieved_context)
        agent_3.save_prompt(prompt, requirements, retrieved_context)
        print(f"Agent 3 prompt synthesized ({len(prompt)} characters).")

        # ----------------------------------------------------------------------
        # STEP 4: MULTIMODAL QUESTION GENERATION (AGENT 4)
        # ----------------------------------------------------------------------
        print("\n>>> [STEP 4/4] Running Agent 4: Multimodal Question Generation...")
        agent_4 = MultimodalChemistryAgent()
        final_count = int(requirements.get("Number", 5))
        final_topic = requirements.get("Topic", "ALL")
        final_diff = requirements.get("Difficulty", "Moderate")

        questions = agent_4.generate(
            total_count=final_count,
            mode=mode,
            image_count=image_count,
            topic=final_topic,
            difficulty=final_diff,
        )

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETED")
        print("=" * 70)
        print(f"Total questions generated : {len(questions)}")
        print(f"Structured JSON output    : {AGENT4_OUTPUT_JSON}")
        print(f"Interactive HTML output   : {AGENT4_OUTPUT_HTML}")
        print(f"Formatted Word DOCX output: {AGENT4_OUTPUT_DOCX}")
        print("=" * 70)

        return questions


def main():
    parser = argparse.ArgumentParser(description="End-to-End Chemistry Multi-Agent Pipeline.")
    parser.add_argument(
        "--request", type=str, default="Generate 5 moderate MCQs on Chemical Kinetics",
        help="Natural language user prompt."
    )
    parser.add_argument(
        "--mode", type=str, default="image", choices=["text", "image", "mixed"],
        help="Question generation mode (text, image, or mixed)."
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Explicit total question count."
    )
    parser.add_argument(
        "--image-count", type=int, default=None,
        help="Explicit number of image questions in mixed mode."
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Explicit topic override."
    )
    parser.add_argument(
        "--difficulty", type=str, default=None,
        help="Explicit difficulty override."
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive CLI mode."
    )
    args = parser.parse_args()

    req_str = args.request
    if args.interactive:
        print("=" * 60)
        print("CHEMISTRY QUESTION GENERATION SYSTEM")
        print("=" * 60)
        req_str = input("Enter your question generation request: ").strip()

    pipeline = ChemistryMultiAgentPipeline()
    pipeline.run(
        user_request=req_str,
        mode=args.mode,
        total_count=args.count,
        image_count=args.image_count,
        difficulty=args.difficulty,
        topic=args.topic,
    )


if __name__ == "__main__":
    main()
