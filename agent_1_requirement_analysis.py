"""
AGENT 1: Requirement Analysis Agent
Parses user request into structured requirements for question generation.
"""

import os
import re
import json
import argparse
from config import AGENT1_REQUIREMENTS_PATH, DATASET_NAME, DATASET_CONFIG

# Default Chemistry broad topics for JEE Main
DEFAULT_TOPICS = [
    "Chemical Bonding and Molecular Structure",
    "Coordination Compounds",
    "Thermodynamics",
    "Chemical Kinetics",
    "Equilibrium",
    "Solutions",
    "Electrochemistry",
    "Atomic Structure",
    "Periodic Classification",
    "p-Block Elements",
    "d- and f-Block Elements",
    "General Organic Chemistry",
    "Hydrocarbons",
    "Haloalkanes and Haloarenes",
    "Alcohols, Phenols and Ethers",
    "Aldehydes, Ketones and Carboxylic Acids",
    "Amines",
    "Biomolecules",
    "Surface Chemistry",
    "States of Matter",
    "Redox Reactions",
    "Solid State",
]

DEFAULT_SUBTOPICS = [
    "VSEPR Theory", "Hybridisation", "Molecular Orbital Theory",
    "Crystal Field Theory", "Isomerism", "IUPAC Nomenclature",
    "First Order Kinetics", "Arrhenius Equation", "Rate Law",
    "Gibbs Free Energy", "Enthalpy", "Entropy", "Hess's Law",
    "Colligative Properties", "Raoult's Law", "Nernst Equation",
    "SN1 and SN2 Reactions", "Electrophilic Aromatic Substitution",
    "Aldol Condensation", "Cannizzaro Reaction", "Carboxylic Acids",
    "Oxidation States", "Coordination Number", "Ligands"
]


class RequirementAnalysisAgent:
    """Agent 1: Extracts structured requirements from natural language prompt."""

    def __init__(self, dataset=None):
        if dataset is not None:
            self.available_topics = sorted(
                set(
                    str(row["topic"]).strip()
                    for row in dataset
                    if row.get("topic") is not None and str(row["topic"]).strip() != ""
                ),
                key=len,
                reverse=True,
            )
            self.available_subtopics = sorted(
                set(
                    str(row["subtopic"]).strip()
                    for row in dataset
                    if row.get("subtopic") is not None and str(row["subtopic"]).strip() != ""
                ),
                key=len,
                reverse=True,
            )
        else:
            self.available_topics = sorted(DEFAULT_TOPICS, key=len, reverse=True)
            self.available_subtopics = sorted(DEFAULT_SUBTOPICS, key=len, reverse=True)

    def analyze(self, user_request: str) -> dict:
        """
        Parses a natural language query into a structured requirements dictionary.
        """
        if not user_request or not user_request.strip():
            return {
                "Topic": "ALL",
                "Question Type": "MCQ",
                "Number": 5,
                "Difficulty": "Moderate"
            }

        request = user_request.lower().strip()

        # 1. EXTRACT NUMBER
        number_match = re.search(r"\b(\d+)\s*(?:questions?|mcqs?|problems?)\b", request)
        if number_match:
            number = int(number_match.group(1))
        else:
            around_match = re.search(r"\b(?:around|about|approximately)\s*(\d+)\b", request)
            if around_match:
                number = int(around_match.group(1))
            else:
                number = 5  # default if not specified

        # 2. EXTRACT QUESTION TYPE
        if re.search(r"\b(mcq|mcqs|multiple choice|objective|single correct)\b", request):
            question_type = "MCQ"
        elif re.search(r"\b(numerical|numericals|integer type|integer)\b", request):
            question_type = "Numerical"
        elif re.search(r"\b(fill in the blank|fill-in-the-blank)\b", request):
            question_type = "Fill in the Blank"
        elif re.search(r"\b(true or false|true/false)\b", request):
            question_type = "True or False"
        else:
            question_type = "MCQ"

        # 3. EXTRACT DIFFICULTY
        if re.search(r"\b(hard|difficult|challenging|tough|advanced)\b", request):
            difficulty = "Hard"
        elif re.search(r"\b(easy|simple|basic)\b", request):
            difficulty = "Easy"
        elif re.search(r"\b(moderate|medium|average|intermediate)\b", request):
            difficulty = "Moderate"
        else:
            difficulty = "Moderate"

        # 4. EXTRACT TOPIC
        topic = None
        if re.search(
            r"\b(all topics|all topic|every topic|every topics|"
            r"all chapters|every chapter|entire syllabus|"
            r"whole syllabus|full syllabus|all chemistry|complete syllabus)\b",
            request,
        ):
            topic = "ALL"
        else:
            for available_topic in self.available_topics:
                if available_topic.lower() in request:
                    topic = available_topic
                    break
            if topic is None:
                for available_subtopic in self.available_subtopics:
                    if available_subtopic.lower() in request:
                        topic = available_subtopic
                        break

        if topic is None:
            topic = "ALL"

        requirements = {
            "Topic": topic,
            "Question Type": question_type,
            "Number": number,
            "Difficulty": difficulty,
        }

        return requirements

    def save_requirements(self, requirements: dict, output_path: str = AGENT1_REQUIREMENTS_PATH):
        """Saves requirement dictionary to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(requirements, f, indent=4)
        print(f"Requirements saved to: {output_path}")


def load_dataset_records():
    """Optional loader to fetch live dataset topics if datasets library is available."""
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_NAME, data_dir=DATASET_CONFIG)
        all_data = ds["train"].to_list() + ds["test"].to_list()
        return all_data
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Agent 1: Requirement Analysis Agent.")
    parser.add_argument(
        "--request", type=str, default="",
        help="Natural language question generation prompt from user."
    )
    parser.add_argument(
        "--output", type=str, default=AGENT1_REQUIREMENTS_PATH,
        help="Path where structured requirements JSON will be saved."
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run interactively in console."
    )
    args = parser.parse_args()

    # Initialize agent
    records = None
    try:
        records = load_dataset_records()
    except Exception:
        pass

    agent_1 = RequirementAnalysisAgent(records)

    req_text = args.request
    if args.interactive or not req_text.strip():
        print("=" * 60)
        print("AGENT 1: REQUIREMENT ANALYSIS AGENT")
        print("=" * 60)
        print("Enter your chemistry question generation request.")
        print("Example: 'Generate 5 moderate MCQs on Chemical Bonding'")
        req_text = input("\nEnter your request: ").strip()

    requirements = agent_1.analyze(req_text)
    print("\n----------------------------------------------")
    print("STRUCTURED REQUIREMENTS:")
    print("----------------------------------------------")
    print(json.dumps(requirements, indent=4))
    print("----------------------------------------------")

    agent_1.save_requirements(requirements, args.output)


if __name__ == "__main__":
    main()
