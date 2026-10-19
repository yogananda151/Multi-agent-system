"""
AGENT 2: Knowledge Graph Retrieval Agent
Retrieves context, reference questions, options, concepts, and images from Knowledge Graph.
"""

import os
import re
import json
import pickle
import random
import difflib
import argparse
import networkx as nx

from config import (
    KG_PICKLE_PATH,
    AGENT1_REQUIREMENTS_PATH,
    AGENT2_OUTPUT_JSON,
    AGENT2_OUTPUT_PKL,
)
from knowledge_graph import ChemistryKnowledgeGraph


def fuzzy_score(a, b):
    """Computes fuzzy string similarity using rapidfuzz if available, else difflib."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b)
    except ImportError:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100


def fuzzy_partial_score(a, b):
    """Computes partial fuzzy similarity."""
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(a, b)
    except ImportError:
        if not a or not b:
            return 0.0
        if a in b or b in a:
            return 100.0
        return difflib.SequenceMatcher(None, a, b).ratio() * 100


def normalize_text(value):
    """Cleans and standardizes text for fuzzy matching."""
    if value is None:
        return ""
    val = str(value).strip().lower()
    val = val.replace("_", " ").replace("-", " ")
    val = re.sub(r"\s+", " ", val)
    return val


class KnowledgeGraphRetrievalAgent:
    """Agent 2: Retrieves relevant chemistry questions and concepts from the Knowledge Graph."""

    def __init__(self, graph=None, kg_path=KG_PICKLE_PATH):
        self.kg_path = kg_path
        if graph is not None:
            self.graph = graph
        else:
            self.graph = self._load_graph()

    def _load_graph(self):
        """Loads graph from pickle file or builds it if absent."""
        if not os.path.exists(self.kg_path):
            print(f"Knowledge Graph not found at {self.kg_path}. Building now...")
            kg_builder = ChemistryKnowledgeGraph(pkl_path=self.kg_path)
            return kg_builder.load_or_build()

        with open(self.kg_path, "rb") as f:
            G = pickle.load(f)
        return G

    def find_subject(self, subject):
        if not subject:
            return None
        norm_target = normalize_text(subject)
        best_node, best_score = None, 0
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Subject":
                score = fuzzy_score(norm_target, normalize_text(data.get("label")))
                if score > best_score and score >= 60:
                    best_score, best_node = score, node
        return best_node

    def find_topic(self, topic):
        if not topic:
            return None
        norm_target = normalize_text(topic)
        if norm_target == "all":
            return None
        best_node, best_score = None, 0
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Topic":
                score = fuzzy_score(norm_target, normalize_text(data.get("label")))
                if score > best_score and score >= 50:
                    best_score, best_node = score, node
        return best_node

    def find_concepts(self, concept):
        if not concept:
            return []
        norm_target = normalize_text(concept)
        matches = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Subtopic":
                label = normalize_text(data.get("label"))
                score = max(fuzzy_score(norm_target, label), fuzzy_partial_score(norm_target, label))
                if score >= 60:
                    matches.append((node, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]

    def retrieve_questions(self, node):
        """Recursively retrieves all downstream Question nodes under a given topic/subtopic."""
        questions = set()
        for successor in self.graph.successors(node):
            node_type = self.graph.nodes[successor].get("type")
            if node_type == "Question":
                questions.add(successor)
            elif node_type == "Subtopic":
                questions.update(self.retrieve_questions(successor))
        return questions

    def get_question_difficulty(self, question_node):
        for neighbor in self.graph.successors(question_node):
            if self.graph.nodes[neighbor].get("type") == "Difficulty":
                return self.graph.nodes[neighbor].get("label")
        return "Moderate"

    def get_question_type(self, question_node):
        return self.graph.nodes[question_node].get("question_type", "MCQ")

    def get_question_id(self, question_node):
        label = self.graph.nodes[question_node].get("label")
        if label:
            return label
        return question_node.replace("QUESTION_", "")

    def get_options(self, question_node):
        """Extracts options A-D with correct flag."""
        options = []
        for neighbor in self.graph.successors(question_node):
            data = self.graph.nodes[neighbor]
            if data.get("type") != "Option":
                continue
            label = str(data.get("label", "")).strip()
            match = re.search(r"([A-D])$", label.upper())
            letter = match.group(1) if match else ""
            text = data.get("text", "")

            edge_data = self.graph.get_edge_data(question_node, neighbor, {})
            is_correct = edge_data.get("relation") == "HAS_CORRECT_OPTION"

            options.append({
                "option": letter,
                "text": text,
                "is_correct": is_correct
            })

        options.sort(key=lambda x: x["option"])
        return options

    def get_images(self, question_node):
        images = []
        for neighbor in self.graph.successors(question_node):
            data = self.graph.nodes[neighbor]
            if data.get("type") != "QuestionImage":
                continue
            images.append({
                "image": data.get("label", ""),
                "path": data.get("path", "")
            })
        return images

    def filter_questions(self, question_nodes, difficulty=None, question_type=None):
        """Filters question nodes by difficulty and question type."""
        filtered = []
        norm_diff = normalize_text(difficulty) if difficulty else None
        norm_type = normalize_text(question_type) if question_type else None

        aliases = {
            "easy": ["easy", "simple", "basic"],
            "moderate": ["moderate", "medium", "average", "intermediate"],
            "hard": ["hard", "difficult", "tough", "advanced"]
        }

        for q in question_nodes:
            q_diff = normalize_text(self.get_question_difficulty(q))
            q_type = normalize_text(self.get_question_type(q))

            diff_match = True
            if norm_diff and norm_diff != "all" and norm_diff != "mixed":
                allowed = aliases.get(norm_diff, [norm_diff])
                diff_match = any(alias in q_diff for alias in allowed)

            type_match = True
            if norm_type:
                if norm_type in ["mcq", "mcqs", "multiple choice"]:
                    type_match = "mcq" in q_type or "single" in q_type or q_type == ""
                elif norm_type in ["numerical", "integer"]:
                    type_match = "numerical" in q_type or "integer" in q_type

            if diff_match and type_match:
                filtered.append(q)

        return filtered

    def rank_questions(self, question_nodes, target_topic=None, target_concept=None, target_difficulty=None, target_type=None):
        """Ranks questions by semantic relevance and hardness."""
        ranked = []
        for q in question_nodes:
            score = 0.0
            q_data = self.graph.nodes[q]
            q_text = normalize_text(q_data.get("question", ""))

            if target_concept:
                score += fuzzy_partial_score(normalize_text(target_concept), q_text) * 0.4
            if target_topic:
                score += fuzzy_partial_score(normalize_text(target_topic), q_text) * 0.3

            q_diff = normalize_text(self.get_question_difficulty(q))
            if target_difficulty and normalize_text(target_difficulty) in q_diff:
                score += 20.0

            ranked.append((q, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in ranked]

    def build_reference_question(self, question_node):
        """Formats Question node into structured reference dictionary."""
        data = self.graph.nodes[question_node]
        question_id = self.get_question_id(question_node)
        question_text = data.get("question", "")
        question_type = self.get_question_type(question_node)
        difficulty = self.get_question_difficulty(question_node)
        options = self.get_options(question_node)
        images = self.get_images(question_node)

        # Trace parents for subtopic, topic, subject
        subtopic, topic, subject = None, None, None
        for parent in self.graph.predecessors(question_node):
            pdata = self.graph.nodes[parent]
            if pdata.get("type") == "Subtopic":
                subtopic = pdata.get("label")
                for tparent in self.graph.predecessors(parent):
                    tdata = self.graph.nodes[tparent]
                    if tdata.get("type") == "Topic":
                        topic = tdata.get("label")
                        for sparent in self.graph.predecessors(tparent):
                            sdata = self.graph.nodes[sparent]
                            if sdata.get("type") == "Subject":
                                subject = sdata.get("label")

        correct_option = None
        for opt in options:
            if opt.get("is_correct"):
                correct_option = opt.get("option")
                break

        return {
            "question_id": question_id,
            "question": question_text,
            "subject": subject or "Chemistry",
            "topic": topic or "General Chemistry",
            "subtopic": subtopic or "General",
            "difficulty": difficulty,
            "question_type": question_type,
            "options": options,
            "correct_option": correct_option,
            "images": images
        }

    def retrieve(self, requirements: dict, output_json=AGENT2_OUTPUT_JSON, output_pkl=AGENT2_OUTPUT_PKL):
        """Main retrieval orchestration method."""
        number = int(requirements.get("Number", 5) or 5)
        topic = requirements.get("Topic")
        concept = requirements.get("Concept")
        difficulty = requirements.get("Difficulty")
        question_type = requirements.get("Question Type") or requirements.get("Type") or "MCQ"

        print("\n" + "=" * 70)
        print("AGENT 2: KNOWLEDGE GRAPH RETRIEVAL")
        print("=" * 70)
        print(f"Target: {number} questions | Topic: {topic} | Difficulty: {difficulty} | Type: {question_type}")

        all_candidate_nodes = []
        is_all_topics = normalize_text(topic) in ["all", "all topics", ""]

        if is_all_topics:
            print("Retrieving representative questions across all syllabus topics...")
            topic_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Topic"]
            for tnode in topic_nodes:
                qnodes = self.retrieve_questions(tnode)
                filtered = self.filter_questions(qnodes, difficulty, question_type)
                if not filtered:
                    filtered = list(qnodes)
                all_candidate_nodes.extend(filtered)
        else:
            topic_node = self.find_topic(topic)
            if topic_node:
                candidate_set = self.retrieve_questions(topic_node)
                all_candidate_nodes = list(candidate_set)
            else:
                # Subtopic / concept search
                concept_nodes = self.find_concepts(topic)
                for cnode in concept_nodes:
                    all_candidate_nodes.extend(list(self.retrieve_questions(cnode)))

            # Filter candidates
            filtered = self.filter_questions(all_candidate_nodes, difficulty, question_type)
            if filtered:
                all_candidate_nodes = filtered

        # If graph has questions, rank them
        fallback_used = False
        if not all_candidate_nodes:
            print("Notice: No exact matches found in KG. Using broad fallback questions.")
            fallback_used = True
            all_candidate_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Question"]

        ranked_nodes = self.rank_questions(
            all_candidate_nodes,
            target_topic=topic,
            target_concept=concept,
            target_difficulty=difficulty,
            target_type=question_type
        )

        selected_nodes = ranked_nodes[:max(number * 4, 20)]
        retrieved_questions = [self.build_reference_question(node) for node in selected_nodes]

        result = {
            "requirements": {
                "Number": number,
                "Exam": requirements.get("Exam", "JEE Main"),
                "Subject": requirements.get("Subject", "Chemistry"),
                "Topic": topic,
                "Concept": concept,
                "Difficulty": difficulty,
                "Type": question_type
            },
            "retrieved_count": len(retrieved_questions),
            "fallback_used": fallback_used,
            "questions": retrieved_questions
        }

        # Save results
        if output_json:
            os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"Retrieved context JSON saved to: {output_json}")

        if output_pkl:
            os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
            with open(output_pkl, "wb") as f:
                pickle.dump(result, f)
            print(f"Retrieved context PKL saved to : {output_pkl}")

        return result


def kg_retrieval_agent(requirements: dict):
    """Convenience wrapper for backward compatibility."""
    agent = KnowledgeGraphRetrievalAgent()
    return agent.retrieve(requirements)


def main():
    parser = argparse.ArgumentParser(description="Agent 2: Knowledge Graph Retrieval Agent.")
    parser.add_argument(
        "--requirements", type=str, default=AGENT1_REQUIREMENTS_PATH,
        help="Path to requirements JSON from Agent 1."
    )
    parser.add_argument(
        "--output-json", type=str, default=AGENT2_OUTPUT_JSON,
        help="Path to output context JSON."
    )
    parser.add_argument(
        "--output-pkl", type=str, default=AGENT2_OUTPUT_PKL,
        help="Path to output context PKL."
    )
    args = parser.parse_args()

    # Load requirements
    if os.path.exists(args.requirements):
        with open(args.requirements, "r", encoding="utf-8") as f:
            reqs = json.load(f)
    else:
        print(f"Notice: Requirements file {args.requirements} not found. Using defaults.")
        reqs = {"Number": 5, "Topic": "Chemical Bonding", "Difficulty": "Moderate", "Question Type": "MCQ"}

    agent = KnowledgeGraphRetrievalAgent()
    agent.retrieve(reqs, output_json=args.output_json, output_pkl=args.output_pkl)


if __name__ == "__main__":
    main()
