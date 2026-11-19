"""
Knowledge Graph Construction and Interactive Visualization
Builds a NetworkX directed graph linking Subjects, Topics, Subtopics, Concepts,
Questions, Options (with correct-option relations), Difficulties, and Images.
"""

import os
import re
import json
import pickle
import shutil
import argparse
import networkx as nx
from config import (
    DATASET_NAME,
    DATASET_CONFIG,
    KG_PICKLE_PATH,
    KG_HTML_PATH,
    IMAGE_ROOT,
    REBUILD_KG_IF_STALE,
)


def normalize_correct_option(value):
    """Convert common dataset encodings (A, 1, option_1, etc.) to A-D."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NAN", "NULL", "EMPTY"}:
        return None
    if text in {"A", "B", "C", "D"}:
        return text
    m = re.search(r"(?:OPTION[_ ]*)?([1-4])\b", text)
    if m:
        return "ABCD"[int(m.group(1)) - 1]
    m = re.search(r"(?:OPTION[_ ]*)?\(?([ABCD])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def propagate_correct_options(graph):
    """
    Ensures bidirectional consistency between Question nodes' correct_option field
    and HAS_CORRECT_OPTION edges.
    1. If a HAS_CORRECT_OPTION edge exists, propagate that option letter to Question node['correct_option'].
    2. If a Question node has a valid correct_option (A-D), ensure the edge to that Option node
       is labeled HAS_CORRECT_OPTION and all other Option edges are HAS_OPTION.
    3. Strips literal 'None' strings from Question nodes.
    """
    if graph is None:
        return graph

    question_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Question"]
    for q in question_nodes:
        q_data = graph.nodes[q]
        curr_correct = normalize_correct_option(q_data.get("correct_option"))

        # Inspect outgoing edges to Option nodes
        edge_correct = None
        option_edges = []
        for succ in graph.successors(q):
            succ_data = graph.nodes.get(succ, {})
            if succ_data.get("type") == "Option":
                lbl = str(succ_data.get("label", "")).strip().upper()
                m = re.search(r"([A-D])$", lbl)
                letter = m.group(1) if m else (succ.split("_")[-1] if succ.split("_")[-1] in "ABCD" else "")
                option_edges.append((succ, letter))

                edge_data = graph.get_edge_data(q, succ) or {}
                rel = str(edge_data.get("relation", "")).upper()
                if (rel == "HAS_CORRECT_OPTION" or "CORRECT_OPTION" in rel) and letter in "ABCD":
                    edge_correct = letter

        # 1. Propagate HAS_CORRECT_OPTION edge -> correct_option node attribute
        if edge_correct and (not curr_correct or curr_correct not in "ABCD"):
            curr_correct = edge_correct
            q_data["correct_option"] = edge_correct
            graph.nodes[q]["correct_option"] = edge_correct

        # 2. Propagate valid correct_option -> HAS_CORRECT_OPTION edge
        if curr_correct in "ABCD":
            q_data["correct_option"] = curr_correct
            graph.nodes[q]["correct_option"] = curr_correct
            for succ, letter in option_edges:
                if letter in "ABCD":
                    new_rel = "HAS_CORRECT_OPTION" if letter == curr_correct else "HAS_OPTION"
                    graph[q][succ]["relation"] = new_rel
                    graph[q][succ]["label"] = new_rel
                    graph[q][succ]["title"] = new_rel
        else:
            # Clean literal string "None"
            q_data["correct_option"] = None
            graph.nodes[q]["correct_option"] = None

    return graph


def as_list(value):
    """Ensure value is parsed into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [text]
    return [value]


def copy_or_materialize_image(image_value, split, row_index, image_index, image_root=IMAGE_ROOT):
    """
    Resolves or materializes an image file from path, dict, or bytes to a local directory.
    """
    os.makedirs(image_root, exist_ok=True)
    raw = image_value
    source_path = None
    raw_bytes = None

    if isinstance(raw, dict):
        source_path = raw.get("path") or raw.get("filename") or raw.get("file")
        raw_bytes = raw.get("bytes")
    else:
        source_path = str(raw) if raw is not None else None

    basename = os.path.basename(str(source_path or f"image_{row_index}_{image_index}.png"))
    ext = os.path.splitext(basename)[1] or ".png"
    target = os.path.join(image_root, f"{split}_{row_index}_{image_index}{ext}")

    if raw_bytes:
        try:
            with open(target, "wb") as f:
                f.write(raw_bytes)
            return target
        except Exception:
            pass

    candidates = []
    if source_path:
        candidates.extend([
            source_path,
            os.path.join(image_root, basename),
            os.path.join(os.path.dirname(image_root), source_path),
            os.path.join(os.path.dirname(image_root), "chemistry", source_path),
        ])

    for candidate in candidates:
        try:
            if candidate and os.path.isfile(candidate):
                shutil.copy2(candidate, target)
                return target
        except Exception:
            continue

    # Attempt download via huggingface_hub if repository-relative path
    try:
        from huggingface_hub import hf_hub_download
        repo_id = DATASET_NAME
        rels = []
        if source_path:
            rels.extend([
                str(source_path).lstrip("/"),
                f"chemistry/{str(source_path).lstrip('/')}",
                f"chemistry/images/{basename}",
                f"images/{basename}",
            ])
        for rel in dict.fromkeys(rels):
            try:
                downloaded = hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset")
                shutil.copy2(downloaded, target)
                return target
            except Exception:
                continue
    except Exception:
        pass

    return None


def is_kg_stale(graph):
    """
    Checks if an existing Knowledge Graph lacks correct-option edges or contains unresolved image nodes.
    Ensures HAS_CORRECT_OPTION edges and correct_option fields are synchronized before evaluation.
    """
    question_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "Question"]
    if not question_nodes:
        return True

    # Synchronize HAS_CORRECT_OPTION edges and correct_option fields
    propagate_correct_options(graph)

    # Validate correct-option edge presence
    has_correct = any(
        graph.get_edge_data(q, n, {}).get("relation") == "HAS_CORRECT_OPTION"
        for q in question_nodes
        for n in graph.successors(q)
    )
    if not has_correct:
        return True

    return False


class ChemistryKnowledgeGraph:
    """Manages building, loading, validating, and visualizing the Chemistry Knowledge Graph."""

    def __init__(self, pkl_path=KG_PICKLE_PATH, html_path=KG_HTML_PATH, image_root=IMAGE_ROOT):
        self.pkl_path = pkl_path
        self.html_path = html_path
        self.image_root = image_root
        self.graph = None

    def build(self, dataset=None):
        """Builds NetworkX Knowledge Graph from the dataset."""
        if dataset is None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Please install datasets via: pip install datasets")
            print(f"Loading '{DATASET_NAME}'...")
            dataset = load_dataset(DATASET_NAME, data_dir=DATASET_CONFIG)

        print("Constructing Knowledge Graph...")
        G = nx.DiGraph()

        for split in dataset:
            for i, item in enumerate(dataset[split]):
                qid = str(item.get("question_id", f"{split}_{i}"))
                question = str(item.get("question", ""))
                subject = str(item.get("subject", "Chemistry"))
                topic = str(item.get("topic", "General Chemistry"))
                subtopic = str(item.get("subtopic", "General"))
                difficulty = str(item.get("difficulty", "Moderate"))
                qtype = str(item.get("question_type", "MCQ"))
                correct = normalize_correct_option(
                    item.get("correct_option") or item.get("correct_answer") or item.get("answer")
                )

                subject_id = f"SUBJECT_{subject}"
                topic_id = f"TOPIC_{topic}"
                subtopic_id = f"SUBTOPIC_{subtopic}"
                question_id = f"QUESTION_{qid}"
                difficulty_id = f"DIFFICULTY_{difficulty}"

                # Add structural nodes
                G.add_node(subject_id, type="Subject", label=subject)
                G.add_node(topic_id, type="Topic", label=topic)
                G.add_node(subtopic_id, type="Subtopic", label=subtopic)
                G.add_node(
                    question_id,
                    type="Question",
                    label=qid,
                    question=question,
                    question_type=qtype,
                    split=split,
                    correct_option=correct,
                )
                G.add_node(difficulty_id, type="Difficulty", label=difficulty)

                # Connect structural nodes
                G.add_edge(subject_id, topic_id, relation="HAS_TOPIC")
                G.add_edge(topic_id, subtopic_id, relation="HAS_SUBTOPIC")
                G.add_edge(subtopic_id, question_id, relation="HAS_QUESTION")
                G.add_edge(question_id, difficulty_id, relation="HAS_DIFFICULTY")

                # Add Options
                options = [
                    ("A", item.get("option_1")),
                    ("B", item.get("option_2")),
                    ("C", item.get("option_3")),
                    ("D", item.get("option_4")),
                ]
                for letter, option_text in options:
                    if option_text is None:
                        continue
                    option_text = str(option_text).strip()
                    if not option_text:
                        continue
                    option_id = f"OPTION_{qid}_{letter}"
                    G.add_node(option_id, type="Option", label=f"Option {letter}", text=option_text)
                    relation = "HAS_CORRECT_OPTION" if correct == letter else "HAS_OPTION"
                    G.add_edge(question_id, option_id, relation=relation)

                # Add Images
                image_values = as_list(item.get("question_images"))
                for j, image_value in enumerate(image_values):
                    if image_value in (None, ""):
                        continue
                    local_path = copy_or_materialize_image(
                        image_value, split, i, j, image_root=self.image_root
                    )
                    if local_path:
                        image_id = f"IMAGE_{qid}_{j+1}"
                        label_name = os.path.basename(
                            str(image_value.get("path", image_value))
                            if isinstance(image_value, dict)
                            else str(image_value)
                        )
                        G.add_node(
                            image_id,
                            type="QuestionImage",
                            label=label_name,
                            path=local_path,
                            source_path=(
                                image_value.get("path")
                                if isinstance(image_value, dict)
                                else str(image_value)
                            ),
                        )
                        G.add_edge(question_id, image_id, relation="HAS_IMAGE")

        propagate_correct_options(G)
        self.graph = G
        self.save_pickle()
        return G

    def save_pickle(self, path=None):
        """Saves current graph as pickle file."""
        target_path = path or self.pkl_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to: {target_path}")

    def load_or_build(self, rebuild_if_stale=REBUILD_KG_IF_STALE):
        """Loads graph if existing & valid, else builds from dataset."""
        if os.path.exists(self.pkl_path):
            print(f"Loading existing Knowledge Graph from: {self.pkl_path}")
            try:
                with open(self.pkl_path, "rb") as f:
                    G = pickle.load(f)
                G = propagate_correct_options(G)
                if not rebuild_if_stale or not is_kg_stale(G):
                    print("Knowledge Graph loaded successfully.")
                    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
                    self.graph = G
                    return G
                print("Existing Knowledge Graph is stale or missing correct edges. Rebuilding...")
            except Exception as e:
                print(f"Failed to load existing graph ({e}). Rebuilding...")

        return self.build()

    def visualize_pyvis(self, output_html=None):
        """Generates interactive PyVis HTML visualization."""
        if self.graph is None:
            self.load_or_build()

        target_html = output_html or self.html_path
        os.makedirs(os.path.dirname(os.path.abspath(target_html)), exist_ok=True)

        try:
            from pyvis.network import Network
        except ImportError:
            print("Notice: 'pyvis' not installed. Skipping interactive visualization.")
            print("To enable visualization: pip install pyvis")
            return

        print(f"Generating interactive PyVis graph: {target_html}")
        net = Network(
            height="850px",
            width="100%",
            directed=True,
            bgcolor="#ffffff",
            font_color="#222222"
        )

        net.set_options("""
        {
          "nodes": { "font": { "size": 14 }, "borderWidth": 1 },
          "edges": { "arrows": { "to": { "enabled": true } }, "smooth": { "enabled": false } },
          "physics": { "enabled": false },
          "interaction": { "hover": true, "navigationButtons": true, "zoomView": true }
        }
        """)

        color_map = {
            "Subject": ("#FF6B6B", 35),
            "Topic": ("#4ECDC4", 30),
            "Subtopic": ("#45B7D1", 25),
            "Question": ("#96CEB4", 18),
            "Option": ("#FFEAA7", 15),
            "Difficulty": ("#DDA0DD", 20),
            "QuestionImage": ("#A29BFE", 20)
        }

        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "")
            label = str(data.get("label", node))
            color, size = color_map.get(node_type, ("#CCCCCC", 15))

            if node_type == "Question":
                q_text = data.get("question", "")
                title = f"<b>Question ID:</b> {label}<br><br>{q_text}"
            elif node_type == "Option":
                opt_text = data.get("text", "")
                title = f"<b>{label}</b><br>{opt_text}"
            else:
                title = f"{node_type}: {label}"

            net.add_node(node, label=label, title=title, color=color, size=size)

        for src, dst, data in self.graph.edges(data=True):
            rel = data.get("relation", "")
            net.add_edge(src, dst, label=rel, title=rel)

        net.save_graph(target_html)
        print("Visualization generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Build and visualize Chemistry Knowledge Graph.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild graph from dataset.")
    parser.add_argument("--visualize", action="store_true", help="Generate PyVis HTML visualization.")
    args = parser.parse_args()

    kg = ChemistryKnowledgeGraph()
    if args.rebuild:
        kg.build()
    else:
        kg.load_or_build()

    if args.visualize:
        kg.visualize_pyvis()


if __name__ == "__main__":
    main()
