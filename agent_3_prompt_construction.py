"""
AGENT 3: Prompt Construction Agent
Constructs the final, structured generation prompt for Agent 4
combining user requirements from Agent 1 and retrieved context from Agent 2.
"""

import os
import json
import argparse
from config import (
    AGENT1_REQUIREMENTS_PATH,
    AGENT2_OUTPUT_JSON,
    AGENT3_OUTPUT_TXT,
    AGENT3_OUTPUT_JSON,
)


class PromptConstructionAgent:
    """Agent 3: Builds detailed generation prompt with few-shot examples and strict formatting constraints."""

    def __init__(self):
        pass

    def build_requirements(self, requirements: dict) -> str:
        """Formats Agent 1 requirements section."""
        number = requirements.get("Number", 5)
        exam = requirements.get("Exam", "JEE Main")
        subject = requirements.get("Subject", "Chemistry")
        topic = requirements.get("Topic", "ALL")
        concept = requirements.get("Concept", "Not specified")
        difficulty = requirements.get("Difficulty", "Moderate")
        question_type = requirements.get("Type", requirements.get("Question Type", "MCQ"))

        text = f"""
TARGET REQUIREMENTS
Subject: {subject}
Exam: {exam}
Target topic: {topic}
Target concept: {concept}
Number of questions: {number}
Difficulty: {difficulty}
Question type: {question_type}
"""
        return text.strip()

    def build_reference_section(self, retrieved_context: dict) -> str:
        """Formats Agent 2 retrieved context questions into few-shot reference examples."""
        if not retrieved_context:
            retrieved_context = {}

        references = retrieved_context.get("questions", [])
        if not references:
            return """
KNOWLEDGE GRAPH REFERENCES
No reference questions were retrieved.
Generate original questions using the specified requirements and JEE Chemistry domain principles.
""".strip()

        lines = ["KNOWLEDGE GRAPH REFERENCES\n"]
        for i, ref in enumerate(references[:15], start=1):
            lines.append(f"REFERENCE QUESTION {i}")
            lines.append(f"Question ID   : {ref.get('question_id', 'N/A')}")
            lines.append(f"Topic         : {ref.get('topic', 'N/A')}")
            lines.append(f"Subtopic      : {ref.get('subtopic', 'N/A')}")
            lines.append(f"Difficulty    : {ref.get('difficulty', 'N/A')}")
            lines.append(f"Question Type : {ref.get('question_type', 'N/A')}")
            lines.append(f"Question:\n{ref.get('question', 'N/A')}")

            options = ref.get("options", [])
            if options:
                lines.append("Options:")
                for opt in options:
                    if isinstance(opt, dict):
                        lines.append(f"({opt.get('option', '')}) {opt.get('text', '')}")
                    else:
                        lines.append(f"{opt}")

            correct = ref.get("correct_option")
            if correct:
                lines.append(f"Correct Option: {correct}")

            lines.append("-" * 40)

        return "\n".join(lines).strip()

    def build_instructions(self, requirements: dict) -> str:
        """Builds generation guidelines according to examination standard and question type."""
        qtype = str(requirements.get("Type", requirements.get("Question Type", "MCQ"))).strip().lower()

        base_instructions = """
GENERATION INSTRUCTIONS
1. Generate new Chemistry questions aligned with the target requirements and referenced Knowledge Graph concepts.
2. The retrieved questions are reference material only. DO NOT copy any reference question directly.
3. Create ORIGINAL questions while preserving scientific correctness, difficulty level, and examination style.
4. Ensure every question is chemically meaningful, unambiguous, and solvable.
5. All chemical equations, formulas, and stoichiometric values must be mathematically and scientifically balanced.
6. Provide comprehensive, step-by-step reasoning in the explanation.
"""

        if any(x in qtype for x in ["mcq", "multiple choice", "objective", "single correct"]):
            format_rules = """
MCQ FORMAT SPECIFICATIONS
- Every question must provide: Question text, Option A, Option B, Option C, Option D, Correct Answer, Explanation.
- Exactly four distinct options (A, B, C, D) must be provided.
- Exactly one option must be unambiguously correct.
- Distractors must be plausible, reflecting common student errors or misconceptions (e.g. inverted ratio, wrong oxidation state).
"""
        elif any(x in qtype for x in ["numerical", "integer"]):
            format_rules = """
NUMERICAL FORMAT SPECIFICATIONS
- Every question must provide: Question text, Numerical Answer, Explanation.
- The numerical question must contain all required constants or given values to compute an exact answer.
- Rounding instructions (e.g. to nearest integer or 2 decimal places) must be clearly stated.
"""
        else:
            format_rules = """
GENERAL FORMAT SPECIFICATIONS
- Every question must have clear problem statement, correct solution, and detailed conceptual explanation.
"""

        return (base_instructions + "\n" + format_rules).strip()

    def build_output_format(self, requirements: dict) -> str:
        """Builds strict JSON response schema."""
        qtype = str(requirements.get("Type", requirements.get("Question Type", "MCQ"))).strip().lower()

        if any(x in qtype for x in ["numerical", "integer"]):
            return """
OUTPUT FORMAT
Return ONLY valid JSON with this exact structure:
{
    "questions": [
        {
            "question": "Problem statement with chemical details...",
            "numerical_answer": "42",
            "explanation": "Step 1: Calculate moles... Step 2: Apply formula..."
        }
    ]
}
""".strip()
        else:
            return """
OUTPUT FORMAT
Return ONLY valid JSON with this exact structure:
{
    "questions": [
        {
            "question": "Question text...",
            "option_A": "First option...",
            "option_B": "Second option...",
            "option_C": "Third option...",
            "option_D": "Fourth option...",
            "correct_answer": "A",
            "explanation": "Detailed explanation showing why A is correct and others are wrong..."
        }
    ]
}
""".strip()

    def build_prompt(self, requirements: dict, retrieved_context: dict) -> str:
        """Assembles the full prompt from components."""
        req_sec = self.build_requirements(requirements)
        ref_sec = self.build_reference_section(retrieved_context)
        inst_sec = self.build_instructions(requirements)
        fmt_sec = self.build_output_format(requirements)

        prompt = f"""
You are a Chemistry question generation system specialized in JEE/NEET-style examination questions.

============================================================
{req_sec}
============================================================
{ref_sec}
============================================================
{inst_sec}
============================================================
{fmt_sec}
============================================================
""".strip()
        return prompt

    def save_prompt(
        self,
        final_prompt: str,
        requirements: dict,
        retrieved_context: dict,
        txt_path: str = AGENT3_OUTPUT_TXT,
        json_path: str = AGENT3_OUTPUT_JSON
    ):
        """Saves prompt text and composite prompt JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final_prompt)
        print(f"Prompt text saved to: {txt_path}")

        prompt_data = {
            "requirements": requirements,
            "retrieved_context": retrieved_context,
            "generation_prompt": final_prompt
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=4, ensure_ascii=False)
        print(f"Prompt JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Agent 3: Prompt Construction Agent.")
    parser.add_argument(
        "--requirements", type=str, default=AGENT1_REQUIREMENTS_PATH,
        help="Path to requirements JSON from Agent 1."
    )
    parser.add_argument(
        "--context", type=str, default=AGENT2_OUTPUT_JSON,
        help="Path to retrieved context JSON from Agent 2."
    )
    parser.add_argument(
        "--output-txt", type=str, default=AGENT3_OUTPUT_TXT,
        help="Path to output prompt text file."
    )
    parser.add_argument(
        "--output-json", type=str, default=AGENT3_OUTPUT_JSON,
        help="Path to output prompt JSON file."
    )
    args = parser.parse_args()

    # Load Agent 1 requirements
    if os.path.exists(args.requirements):
        with open(args.requirements, "r", encoding="utf-8") as f:
            reqs = json.load(f)
    else:
        reqs = {"Topic": "Chemical Bonding", "Question Type": "MCQ", "Number": 5, "Difficulty": "Moderate"}

    # Load Agent 2 context
    if os.path.exists(args.context):
        with open(args.context, "r", encoding="utf-8") as f:
            ctx = json.load(f)
    else:
        ctx = {"questions": []}

    agent_3 = PromptConstructionAgent()
    final_prompt = agent_3.build_prompt(reqs, ctx)
    agent_3.save_prompt(final_prompt, reqs, ctx, txt_path=args.output_txt, json_path=args.output_json)

    print("\n" + "=" * 70)
    print("PROMPT CONSTRUCTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
