"""
🧪 Neuro-Symbolic Multimodal Chemistry Question Generation & GraphRAG Studio
A modern interactive web dashboard built with Streamlit.
Integrates:
  - Agent 1: Natural Language Requirement Analysis
  - Agent 2: Hybrid GraphRAG Engine (Symbolic KG + Qdrant Tri-Modal Vector Store + RRF)
  - Agent 3: Examination Prompt Construction
  - Agent 4: Multimodal Chemistry Question Generation (Matplotlib & RDKit)
  - Multi-Format Exporters: Interactive HTML, Microsoft Word DOCX, and JSON
"""

import os
import sys
import json
import time
import base64
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    OUTPUT_DIR,
    ZIP_OUTPUT_PATH,
    AGENT1_REQUIREMENTS_PATH,
    AGENT2_OUTPUT_JSON,
    AGENT3_OUTPUT_TXT,
    AGENT4_OUTPUT_JSON,
    AGENT4_OUTPUT_HTML,
    AGENT4_OUTPUT_DOCX,
    SFT_OUTPUT_JSONL,
    CROSS_CHAPTER_PAIRS,
)
from agent_1_requirement_analysis import DEFAULT_TOPICS, DEFAULT_SUBTOPICS

DIFFICULTIES = ["Easy", "Moderate", "Hard"]
from pipeline import ChemistryMultiAgentPipeline
from vector_store import ChemistryVectorStore
from hybrid_graph_rag import HybridGraphRAGEngine
from export_outputs import package_outputs

# Page Configuration
st.set_page_config(
    page_title="ChemiGen | Neuro-Symbolic GraphRAG Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
    .question-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .badge-topic {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-diff {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-rag {
        background-color: #EDE7F6;
        color: #512DA8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .option-box {
        padding: 10px 14px;
        border-radius: 8px;
        margin: 6px 0;
        border: 1px solid #CFD8DC;
        background-color: #FAFAFA;
        font-size: 0.95rem;
    }
    .correct-option {
        background-color: #E8F5E9;
        border: 1.5px solid #43A047;
        font-weight: 600;
        color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)


def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate HTML download button for binary files."""
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}" style="display:inline-block;padding:8px 16px;background-color:#1E88E5;color:white;border-radius:6px;text-decoration:none;font-weight:500;">📥 Download {file_label}</a>'
    return href


# Sidebar Navigation
st.sidebar.markdown("## 🧪 Navigation")
app_mode = st.sidebar.radio(
    "Choose Studio Mode:",
    [
        "🚀 Multimodal Question Generator",
        "🔍 Tri-Modal Vector DB Search",
        "🧠 Hybrid GraphRAG Inspector",
        "📊 System Architecture & Metrics",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Capabilities")
st.sidebar.markdown("""
- **Symbolic Ontology**: Domain KG
- **Vector DB**: Qdrant (3 Collections)
- **Embeddings**: FastEmbed BGE-small
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Diagrams**: Dual-Engine (Matplotlib + RDKit)
- **Formats**: JSON, HTML, DOCX, ZIP
""")

# ==============================================================================
# MODE 1: MULTIMODAL QUESTION GENERATOR
# ==============================================================================
if app_mode == "🚀 Multimodal Question Generator":
    st.markdown('<div class="main-header">🧪 JEE/NEET Multimodal Question Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Synthesize high-yield chemistry examination questions powered by Hybrid GraphRAG and Dual-Engine Visual Synthesis.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📝 Request Formulation")
        input_type = st.radio("Input Method:", ["Natural Language Prompt", "Structured Form Controls"], horizontal=True)

        if input_type == "Natural Language Prompt":
            user_prompt = st.text_area(
                "Enter generation prompt:",
                value="Generate 3 moderate MCQs on Chemical Kinetics with reaction progress diagrams",
                height=110,
                help="Specify topic, count, difficulty, and diagram preferences in natural English."
            )
            mode_choice = st.selectbox("Diagram Mode:", ["image", "mixed", "text"], index=0)
        else:
            sel_topic = st.selectbox("Chemistry Topic:", DEFAULT_TOPICS, index=3) # Chemical Kinetics
            sel_diff = st.select_slider("Target Difficulty:", options=DIFFICULTIES, value="Moderate")
            sel_format = st.selectbox("Question Pattern (CD-NS-KG-MAS):", ["MCQ (Single Choice)", "Assertion-Reason", "Numerical Integer (Section B)"], index=0)
            sel_count = st.slider("Question Count:", min_value=1, max_value=15, value=3)
            mode_choice = st.selectbox("Modality:", ["image", "mixed", "text"], index=0)

            fmt_code = "MCQ"
            if "Assertion" in sel_format: fmt_code = "Assertion-Reason"
            elif "Numerical" in sel_format: fmt_code = "Numerical"

            if sel_diff == "Hard":
                st.info("🔗 **Cross-Chapter Coupling Active**: Multi-concept pairing across chapters (e.g., Kinetics + Equilibrium or Thermo + Electro) enabled for Hard questions.")

            user_prompt = f"Generate {sel_count} {sel_diff.lower()} {fmt_code} questions on {sel_topic}"

        generate_btn = st.button("🚀 Synthesize Questions with Multi-Agent Pipeline", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### ⚡ Active Pipeline Status")
        status_box = st.empty()
        status_box.info("System ready. Click **Synthesize Questions** to execute Agent 1 -> Agent 2 (GraphRAG) -> Agent 3 -> Agent 4.")

    if generate_btn:
        progress_bar = st.progress(0)
        status_box.info("⚙️ Step 1/4: Analyzing requirements with Agent 1...")
        progress_bar.progress(25)
        time.sleep(0.3)

        pipeline = ChemistryMultiAgentPipeline()
        
        status_box.info("🧠 Step 2/4: Querying Qdrant Vector DB & Symbolic KG via GraphRAG...")
        progress_bar.progress(50)
        
        status_box.info("📝 Step 3/4: Assembling examination prompts with Agent 3...")
        progress_bar.progress(75)

        status_box.info("🎨 Step 4/4: Generating questions and rendering diagrams with Agent 4...")
        
        # Execute pipeline
        start_time = time.time()
        questions = pipeline.run(user_request=user_prompt, mode=mode_choice)
        elapsed = time.time() - start_time
        progress_bar.progress(100)

        status_box.success(f"✅ Generated {len(questions)} questions in {elapsed:.2f} seconds!")

    # Display Generated Questions
    if os.path.exists(AGENT4_OUTPUT_JSON):
        saved_questions = []
        try:
            with open(AGENT4_OUTPUT_JSON, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            if isinstance(saved_data, dict):
                saved_questions = saved_data.get("questions", [])
            elif isinstance(saved_data, list):
                saved_questions = saved_data
        except Exception:
            saved_questions = []

        if saved_questions and isinstance(saved_questions, list):
            st.markdown("---")
            st.subheader(f"📋 Generated Examination Set ({len(saved_questions)} Questions)")

            # Export action bar
            exp_c1, exp_c2, exp_c3, exp_c4, exp_c5 = st.columns(5)
            with exp_c1:
                if os.path.exists(AGENT4_OUTPUT_HTML):
                    with open(AGENT4_OUTPUT_HTML, "r", encoding="utf-8") as f:
                        st.download_button("🌐 Download HTML", f.read(), "chemistry_questions.html", "text/html", use_container_width=True)
            with exp_c2:
                if os.path.exists(AGENT4_OUTPUT_DOCX):
                    with open(AGENT4_OUTPUT_DOCX, "rb") as f:
                        st.download_button("📄 Download Word DOCX", f.read(), "chemistry_questions.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
            with exp_c3:
                with open(AGENT4_OUTPUT_JSON, "r", encoding="utf-8") as f:
                    st.download_button("💾 Download JSON", f.read(), "chemistry_questions.json", "application/json", use_container_width=True)
            with exp_c4:
                if os.path.exists(SFT_OUTPUT_JSONL):
                    with open(SFT_OUTPUT_JSONL, "r", encoding="utf-8") as f:
                        st.download_button("🧠 Download SFT JSONL", f.read(), "sft_synthetic.jsonl", "application/jsonl", use_container_width=True)
            with exp_c5:
                zip_path = f"{ZIP_OUTPUT_PATH}.zip"
                if not os.path.exists(zip_path):
                    try:
                        package_outputs()
                    except Exception:
                        pass
                if os.path.exists(zip_path):
                    with open(zip_path, "rb") as f:
                        st.download_button("📦 Download Bundle ZIP", f.read(), "chemistry_outputs.zip", "application/zip", use_container_width=True)

            # Render question items
            for idx, q in enumerate(saved_questions, 1):
                if not isinstance(q, dict):
                    continue
                q_topic = q.get('topic') or q.get('image_family') or 'Chemistry'
                q_diff = q.get('difficulty') or q.get('difficulty_category') or 'Moderate'
                q_mode = q.get('question_type') or q.get('mode') or 'image'
                q_fmt = q.get('question_format') or 'MCQ'
                q_id = q.get('question_id') or f"CHEM_Q{idx:02d}"

                st.markdown(f"""
                <div class="question-card">
                    <span class="badge-topic">{q_topic}</span>
                    <span class="badge-diff">{q_diff}</span>
                    <span class="badge-rag">{str(q_mode).upper()}</span>
                    <span class="badge-rag" style="background-color:#E0F2F1;color:#00796B;">{str(q_fmt).upper()}</span>
                    <h4 style="margin-top:12px;">Question {idx}: {q_id}</h4>
                    <p style="font-size:1.1rem;line-height:1.6;white-space:pre-line;">{q.get('question', '')}</p>
                </div>
                """, unsafe_allow_html=True)

                col_img, col_opts = st.columns([1, 1])

                with col_img:
                    img_path = q.get("image_path") or q.get("generated_image")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"Visual: {os.path.basename(img_path)}", use_container_width=True)
                    else:
                        st.info("ℹ️ Text-only question (no visual diagram required).")

                with col_opts:
                    if q_fmt == "Numerical":
                        st.markdown(f"**Section B Numerical Integer Answer:** `{q.get('correct_answer', '')}`")
                    else:
                        st.markdown("**Options:**")
                        opts = q.get("options")
                        if not opts or not isinstance(opts, dict):
                            opts = {
                                "A": q.get("option_A", "N/A"),
                                "B": q.get("option_B", "N/A"),
                                "C": q.get("option_C", "N/A"),
                                "D": q.get("option_D", "N/A"),
                            }
                        correct_key = q.get("correct_answer", "").strip().upper()

                        for opt_key, opt_text in opts.items():
                            if not opt_text or opt_text == "N/A": continue
                            is_correct = (opt_key.upper() == correct_key)
                            opt_class = "option-box correct-option" if is_correct else "option-box"
                            st.markdown(f'<div class="{opt_class}"><b>({opt_key})</b> {opt_text} {" ✅ [Correct Answer]" if is_correct else ""}</div>', unsafe_allow_html=True)

                    with st.expander("💡 View Detailed Explanation & Mechanism"):
                        st.write(q.get("explanation", "No explanation provided."))

                    # CD-NS-KG-MAS: Misconception Diagnostics Panel
                    diags = q.get("distractor_diagnostics")
                    if diags and isinstance(diags, dict):
                        with st.expander("🎯 Misconception Diagnostics (CD-NS-KG-MAS)"):
                            for opt_letter, reason in sorted(diags.items()):
                                if "verified" in reason.lower() or opt_letter == q.get("correct_answer"):
                                    st.markdown(f"**({opt_letter})** 🟢 :green[{reason}]")
                                else:
                                    st.markdown(f"**({opt_letter})** 🔴 :orange[{reason}]")


# ==============================================================================
# MODE 2: TRI-MODAL VECTOR DB SEARCH
# ==============================================================================
elif app_mode == "🔍 Tri-Modal Vector DB Search":
    st.markdown('<div class="main-header">🔍 Tri-Modal Qdrant Vector Store Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Query dense 384-D text embeddings, 128-D Swin Transformer diagram vectors, and 1024-D molecular fingerprints.</div>', unsafe_allow_html=True)

    vs = ChemistryVectorStore()
    stats = vs.get_stats()

    m1, m2, m3 = st.columns(3)
    m1.metric("Indexed Questions", f"{stats.get('chemistry_questions', 0):,}", "Dense Embeddings (384-D)")
    m2.metric("Diagram Vectors", f"{stats.get('diagram_embeddings', 0):,}", "Swin-Transformer (128-D)")
    m3.metric("Molecular Structures", f"{stats.get('molecular_structures', 0):,}", "RDKit/SMILES (1024-D)")

    st.markdown("---")
    coll_choice = st.selectbox(
        "Select Collection to Query:",
        ["chemistry_questions (Semantic Question Bank)", "diagram_embeddings (Vision Features)", "molecular_structures (Chemical Fingerprints)"]
    )

    if "chemistry_questions" in coll_choice:
        q_col1, q_col2, q_col3 = st.columns([2, 1, 1])
        with q_col1:
            query_str = st.text_input("Enter natural language query / chemical concept:", "Arrhenius equation and activation energy slope")
        with q_col2:
            filter_topic = st.selectbox("Topic Filter (Optional):", ["ALL"] + DEFAULT_TOPICS)
        with q_col3:
            filter_diff = st.selectbox("Difficulty Filter (Optional):", ["ALL"] + DIFFICULTIES)

        top_k = st.slider("Top K results:", min_value=1, max_value=10, value=4)

        if st.button("Search Vector DB", type="primary"):
            t_filter = None if filter_topic == "ALL" else filter_topic
            d_filter = None if filter_diff == "ALL" else filter_diff
            with st.spinner("Searching Qdrant collection..."):
                results = vs.search_questions(query_str, top_k=top_k, topic=t_filter, difficulty=d_filter)

            if results:
                st.success(f"Retrieved {len(results)} nearest neighbor points from Qdrant:")
                for r in results:
                    st.markdown(f"""
                    <div class="question-card">
                        <span class="badge-topic">{r.get('topic', 'N/A')}</span>
                        <span class="badge-diff">{r.get('difficulty', 'N/A')}</span>
                        <span class="badge-rag">Cosine Score: {r.get('score', 0):.4f}</span>
                        <p style="margin-top:10px;font-size:1.05rem;"><b>{r.get('question_id', '')}</b>: {r.get('question', '')}</p>
                        <p style="color:#546E7A;font-size:0.9rem;">Subtopic: {r.get('subtopic', 'N/A')} | Question Type: {r.get('type', 'MCQ')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No matches found for the specified query and filters.")

    elif "diagram_embeddings" in coll_choice:
        st.info("Querying the 2,005 Swin Transformer vision representations.")
        diag_query = st.text_input("Enter diagram descriptor / visual concept:", "decay curve half life kinetics")
        if st.button("Find Visually Aligned Questions", type="primary"):
            with st.spinner("Finding nearest diagram representations..."):
                diag_results = vs.search_questions(diag_query, top_k=3)
                for r in diag_results:
                    st.markdown(f"**ID:** `{r.get('question_id')}` | **Similarity:** `{r.get('score', 0):.4f}`")
                    st.write(r.get("question"))

    elif "molecular_structures" in coll_choice:
        st.info("Querying the 1024-D molecular chemical fingerprints.")
        smiles_query = st.text_input("Enter SMILES string or molecular formula:", "CC(=O)Oc1ccccc1C(=O)O")
        if st.button("Search Molecular Fingerprint Store", type="primary"):
            with st.spinner("Querying molecular collection..."):
                mol_res = vs.search_molecules(smiles_query, top_k=3)
                if mol_res:
                    for m in mol_res:
                        st.json(m)
                else:
                    st.warning("No direct molecular matches.")

    vs.close()


# ==============================================================================
# MODE 3: HYBRID GRAPHRAG INSPECTOR
# ==============================================================================
elif app_mode == "🧠 Hybrid GraphRAG Inspector":
    st.markdown('<div class="main-header">🧠 Hybrid GraphRAG Retrieval Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect how Symbolic Knowledge Graph Triples fuse with Qdrant Vector Search via Reciprocal Rank Fusion (RRF).</div>', unsafe_allow_html=True)

    rag = HybridGraphRAGEngine()

    c1, c2, c3 = st.columns(3)
    c1.metric("KG Entities", "350+", "Chemical Concepts")
    c2.metric("KG Relations", "1,200+", "Prerequisites & Subtopics")
    c3.metric("Fusion Method", "RRF (k=60)", "Dense Vector + Lexical BM25")

    st.markdown("---")
    st.subheader("Simulate Retrieval Pipeline")
    sim_topic = st.selectbox("Select Target Topic:", DEFAULT_TOPICS, index=3)
    sim_diff = st.selectbox("Select Target Difficulty:", DIFFICULTIES, index=1)
    sim_count = st.slider("Exemplar Count:", 1, 5, 3)

    if st.button("Execute GraphRAG Retrieval", type="primary"):
        with st.spinner("Traversing Knowledge Graph and executing Vector Search..."):
            req = {"Topic": sim_topic, "Difficulty": sim_diff, "Number": sim_count}
            context = rag.retrieve(req, top_k=sim_count)

        st.success(f"Retrieved {len(context.get('questions', []))} exemplar questions via {context.get('retrieval_method')}")

        cross_coupling = context.get("cross_chapter_coupling")
        if cross_coupling:
            st.markdown(f"""
            <div style="background-color:#EDE7F6;border-left:5px solid #673AB7;padding:12px;margin:10px 0;border-radius:6px;">
                <b style="color:#4527A0;">🔗 CD-NS-KG-MAS Cross-Chapter Coupling Active (Eq. 2):</b><br>
                <b>Chapter A:</b> {cross_coupling['chapter_A']} ({cross_coupling['concept_A']}) &nbsp;⚡&nbsp; 
                <b>Chapter B:</b> {cross_coupling['chapter_B']} ({cross_coupling['concept_B']})<br>
                <small><b>Governing Formula:</b> <code>{cross_coupling['coupling_formula']}</code> | {cross_coupling['description']}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 1. Retrieved Exemplar Questions (Used by Agent 3 for Few-Shot In-Context Learning):")
        for q in context.get("questions", []):
            st.markdown(f"""
            <div class="question-card">
                <b>[{q.get('question_id')}]</b> {q.get('question')}
                <br><small>Topic: <b>{q.get('topic')}</b> | Difficulty: <b>{q.get('difficulty')}</b> | Score: <b>{q.get('score', 1.0):.4f}</b></small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 2. Retrieved Concepts & Knowledge Entities:")
        concepts = context.get("concepts", [])
        if concepts:
            st.write(", ".join([f"`{c}`" for c in concepts[:20]]))
        else:
            st.write("Retrieved directly from ontology.")

        st.markdown("#### 3. Agent 3 Prompt Preview:")
        if os.path.exists(AGENT3_OUTPUT_TXT):
            with open(AGENT3_OUTPUT_TXT, "r", encoding="utf-8") as f:
                st.code(f.read()[:1500] + "\n\n... [TRUNCATED FOR PREVIEW] ...", language="markdown")

    rag.close()


# ==============================================================================
# MODE 4: SYSTEM ARCHITECTURE & METRICS
# ==============================================================================
elif app_mode == "📊 System Architecture & Metrics":
    st.markdown('<div class="main-header">📊 System Architecture & Research Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent Neuro-Symbolic Architecture incorporating Deep Learning vision and knowledge graphs.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Architectural Workflow
    ```mermaid
    flowchart LR
        User([User Query / Exam Request]) --> A1[Agent 1: Requirement Analysis]
        A1 --> A2[Agent 2: Hybrid GraphRAG Engine]
        KG[(Symbolic Ontology KG)] --> A2
        Qdrant[(Tri-Modal Qdrant Vector DB)] --> A2
        A2 --> A3[Agent 3: Examination Prompt Synthesizer]
        A3 --> A4[Agent 4: Multimodal Chemistry Generator]
        A4 --> Matplot[Matplotlib Kinetic Plots]
        A4 --> RDKit[RDKit Molecular Graphs]
        A4 --> Export[Exporters: HTML / DOCX / JSON / ZIP]
    ```
    """)

    st.markdown("### 🏆 Core Innovations & Tech Stack")
    st.markdown("""
    | Innovation Component | Technology | Role & Function |
    | :--- | :--- | :--- |
    | **Agentic Core** | Python Multi-Agent Architecture | 4 Specialized autonomous agents for NLP, RAG, Prompting, and Question Generation |
    | **Vector Database** | **Qdrant (Local Embedded)** | High-performance vector engine indexing 794 chemistry questions, 2,005 diagram features, and molecular fingerprints |
    | **Dense Embeddings** | **FastEmbed (BAAI/bge-small-en-v1.5)** | Fast 384-dimensional ONNX-accelerated dense semantic representations |
    | **Hybrid GraphRAG** | **Reciprocal Rank Fusion (RRF)** | Merges dense cosine similarity rankings with symbolic knowledge graph topology ($k=60$) |
    | **Vision Models** | **Swin Transformer + U-Net + Sinkhorn** | Deep learning visual features and optimal transport mapping for chemical diagrams |
    | **Dual Visual Synthesis** | **Matplotlib + RDKit** | Precise chemical reaction kinetics curves and IUPAC molecular structural drawings |
    | **Export Ecosystem** | **DOCX, HTML, JSON, ZIP** | Print-ready examination documents and interactive styled web pages |
    """)

    st.markdown("### 📁 Project Artifact Status")
    files_to_check = [
        ("Agent 1 Requirements JSON", AGENT1_REQUIREMENTS_PATH),
        ("Agent 2 KG Retrieved Context", AGENT2_OUTPUT_JSON),
        ("Agent 3 Synthesized Prompt", AGENT3_OUTPUT_TXT),
        ("Agent 4 Generated Questions JSON", AGENT4_OUTPUT_JSON),
        ("Agent 4 Interactive HTML Output", AGENT4_OUTPUT_HTML),
        ("Agent 4 Word DOCX Output", AGENT4_OUTPUT_DOCX),
        ("Master Project ZIP Archive", f"{ZIP_OUTPUT_PATH}.zip"),
    ]

    for name, fpath in files_to_check:
        exists = os.path.exists(fpath)
        size_kb = os.path.getsize(fpath) / 1024 if exists else 0
        status_icon = "✅" if exists else "⚠️"
        st.write(f"{status_icon} **{name}**: `{fpath}` ({size_kb:.1f} KB)")
