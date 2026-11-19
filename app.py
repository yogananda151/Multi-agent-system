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
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    ZIP_OUTPUT_PATH,
    GENERATED_IMAGES_DIR,
    IMAGE_ROOT,
    AGENT1_REQUIREMENTS_PATH,
    AGENT2_OUTPUT_JSON,
    AGENT3_OUTPUT_TXT,
    AGENT4_OUTPUT_JSON,
    AGENT4_OUTPUT_HTML,
    AGENT4_OUTPUT_DOCX,
    SFT_OUTPUT_JSONL,
)
from agent_1_requirement_analysis import DEFAULT_TOPICS

DIFFICULTIES = ["Easy", "Moderate", "Hard"]
from pipeline import ChemistryMultiAgentPipeline
from vector_store import ChemistryVectorStore
from hybrid_graph_rag import HybridGraphRAGEngine
from export_outputs import package_outputs
from vision_engine import (
    DiagramRestorationEngine,
    DiagramClassifier,
    ReverseDiagramSearchEngine,
)

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


# Sidebar Navigation
st.sidebar.markdown("## 🧪 Navigation")
app_mode = st.sidebar.radio(
    "Choose Studio Mode:",
    [
        "🚀 Multimodal Question Generator",
        "📸 Fullstack Computer Vision Studio",
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
- **Computer Vision**: U-Net & Swin-T
- **Embeddings**: FastEmbed BGE-small
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Diagrams**: Dual-Engine (Matplotlib + RDKit)
- **Formats**: JSON, HTML, DOCX, SFT JSONL, ZIP
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
# MODE 2: FULLSTACK COMPUTER VISION STUDIO
# ==============================================================================
elif app_mode == "📸 Fullstack Computer Vision Studio":
    st.markdown('<div class="main-header">📸 Fullstack Computer Vision Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Vision Suite: PyTorch U-Net Restoration, Swin Transformer 128-D Feature Extraction, Reverse Visual Search, & Synthetic Diagram Lab.</div>', unsafe_allow_html=True)

    cv_tab1, cv_tab2, cv_tab3, cv_tab4 = st.tabs([
        "🔬 U-Net Diagram Denoising & Restoration",
        "🔍 Reverse Visual Diagram Search (CBIR)",
        "🏷️ Visual Diagram Anatomy & Classifier",
        "🎨 Synthetic Diagram Laboratory",
    ])

    # Sample images finder
    available_samples = []
    for folder in [GENERATED_IMAGES_DIR, IMAGE_ROOT]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    available_samples.append(os.path.join(folder, f))

    # --------------------------------------------------------------------------
    # TAB 1: U-NET DIAGRAM RESTORATION
    # --------------------------------------------------------------------------
    with cv_tab1:
        st.markdown("#### 🔬 Neural Diagram Restoration & Smudge Removal (PyTorch U-Net)")
        st.markdown("Cleans scanned historical examination diagrams, eliminates background noise, and reconstructs crisp chemical bonds.")

        u_col1, u_col2 = st.columns([1, 1])
        with u_col1:
            input_method = st.radio("Select Image Source:", ["Choose Sample Diagram from Project", "Upload Custom Diagram (PNG/JPG)"], horizontal=True)
            chosen_img = None
            if input_method == "Choose Sample Diagram from Project" and available_samples:
                sel_sample = st.selectbox("Select Sample Diagram:", available_samples, format_func=lambda p: os.path.basename(p))
                if sel_sample and os.path.exists(sel_sample):
                    chosen_img = Image.open(sel_sample)
            else:
                uploaded = st.file_uploader("Upload scanned/degraded diagram:", type=["png", "jpg", "jpeg"])
                if uploaded:
                    chosen_img = Image.open(uploaded)

            if chosen_img:
                st.image(chosen_img, caption="Input Image Preview", use_container_width=True)

        with u_col2:
            st.markdown("##### 🎛️ Neural Restoration Hyperparameters")
            contrast_val = st.slider("Contrast Enhancement Factor:", 1.0, 3.0, 1.4, 0.1)
            sharp_val = st.slider("Unsharp Mask Line Sharpness:", 1.0, 3.0, 1.5, 0.1)
            binarize_chk = st.checkbox("Apply Adaptive Document Line Binarization", value=False)

            if chosen_img and st.button("🚀 Execute PyTorch U-Net Restoration", type="primary", use_container_width=True):
                with st.spinner("Processing through U-Net Encoder-Bottleneck-Decoder..."):
                    restorer = DiagramRestorationEngine()
                    restored_img, metrics = restorer.denoise_and_restore(
                        chosen_img,
                        contrast_boost=contrast_val,
                        sharpness=sharp_val,
                        threshold_binarize=binarize_chk
                    )

                st.success("✅ Diagram Restored Successfully!")
                st.image(restored_img, caption="Restored Output (U-Net Denoised)", use_container_width=True)

                st.markdown("##### 📊 Image Restoration Metrics:")
                rm1, rm2, rm3 = st.columns(3)
                rm1.metric("Resolution", metrics["original_dimensions"])
                rm2.metric("Contrast Improvement", f"+{metrics['contrast_improvement_pct']}%")
                rm3.metric("Engine Device", metrics["device_used"].upper())

                buf = io.BytesIO()
                restored_img.save(buf, format="PNG")
                st.download_button("📥 Download Restored PNG", buf.getvalue(), "restored_diagram.png", "image/png", use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: REVERSE DIAGRAM SEARCH (CBIR)
    # --------------------------------------------------------------------------
    with cv_tab2:
        st.markdown("#### 🔍 Reverse Visual Diagram Search (Content-Based Image Retrieval)")
        st.markdown("Drop in any kinetics curve or molecular graph: the **Swin Transformer Tiny** computes its 128-D visual feature representation and retrieves matching past-year questions from Qdrant.")

        s_col1, s_col2 = st.columns([1, 1])
        search_img = None
        with s_col1:
            if available_samples:
                sel_search_sample = st.selectbox("Select Query Diagram:", available_samples, key="search_sample", format_func=lambda p: os.path.basename(p))
                if sel_search_sample and os.path.exists(sel_search_sample):
                    search_img = Image.open(sel_search_sample)
            search_upload = st.file_uploader("Or upload image to search:", type=["png", "jpg", "jpeg"], key="search_upload")
            if search_upload:
                search_img = Image.open(search_upload)

            if search_img:
                st.image(search_img, caption="Query Visual", use_container_width=True)

        with s_col2:
            st.markdown("##### ⚡ Visual Representation & Nearest Neighbors")
            top_k_diagrams = st.slider("Top Matched Exam Diagrams:", 1, 6, 3)
            if search_img and st.button("🔍 Extract Swin-T Embedding & Query Vector DB", type="primary", use_container_width=True):
                with st.spinner("Extracting 128-D Swin Transformer representation & querying Qdrant..."):
                    search_engine = ReverseDiagramSearchEngine()
                    matches = search_engine.search_by_image(search_img, top_k=top_k_diagrams)
                    search_engine.close()

                if matches:
                    st.success(f"Retrieved {len(matches)} visually matching exam questions!")
                    for idx, m in enumerate(matches, 1):
                        st.markdown(f"""
                        <div class="question-card">
                            <span class="badge-topic">{m.get('topic', 'Chemistry')}</span>
                            <span class="badge-rag">Swin-T Cosine Score: {m.get('similarity_score', 0):.4f}</span>
                            <br><b>Match #{idx}:</b> <code>{m.get('filename')}</code>
                            <p style="margin-top:8px;">{m.get('question')}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 3: DIAGRAM ANATOMY CLASSIFIER
    # --------------------------------------------------------------------------
    with cv_tab3:
        st.markdown("#### 🏷️ Visual Diagram Anatomy & Geometry Classifier")
        st.markdown("Deep visual feature analysis to classify coordinate axes, edge distributions, and chemical diagram families.")

        c_col1, c_col2 = st.columns([1, 1])
        class_img = None
        with c_col1:
            if available_samples:
                sel_class_sample = st.selectbox("Select Diagram to Classify:", available_samples, key="class_sample", format_func=lambda p: os.path.basename(p))
                if sel_class_sample and os.path.exists(sel_class_sample):
                    class_img = Image.open(sel_class_sample)
            if class_img:
                st.image(class_img, caption="Diagram for Anatomy Inspection", use_container_width=True)

        with c_col2:
            if class_img and st.button("🏷️ Run AI Visual Classifier", type="primary", use_container_width=True):
                classifier = DiagramClassifier()
                c_res = classifier.classify(class_img)

                st.markdown(f"""
                <div class="question-card" style="border-left:5px solid #1E88E5;">
                    <h3 style="margin:0;color:#1565C0;">{c_res['predicted_family']}</h3>
                    <p style="margin-top:6px;font-size:1.1rem;"><b>Confidence:</b> {int(c_res['confidence'] * 100)}%</p>
                    <p style="color:#546E7A;"><b>Anatomy Rationale:</b> {c_res['reason']}</p>
                </div>
                """, unsafe_allow_html=True)

                cm1, cm2, cm3 = st.columns(3)
                cm1.metric("Cartesian Axes", "Detected" if c_res["has_axes"] else "None (Molecule)")
                cm2.metric("Edge Density", f"{c_res['edge_density']}")
                cm3.metric("Aspect Ratio", f"{c_res['aspect_ratio']}")

    # --------------------------------------------------------------------------
    # TAB 4: SYNTHETIC DIAGRAM LABORATORY
    # --------------------------------------------------------------------------
    with cv_tab4:
        st.markdown("#### 🎨 Interactive Synthetic Diagram Laboratory")
        st.markdown("Live mathematical curve synthesis for chemical reaction kinetics and Arrhenius temperature dependency.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lab_col1, lab_col2 = st.columns([1, 1])

        with lab_col1:
            plot_type = st.selectbox("Select Curve Family:", [
                "Arrhenius Activation Energy (ln k vs 1/T)",
                "First-Order Reaction Decay (ln[A] vs Time)",
                "Zero-Order Reaction Decay ([A] vs Time)",
            ])

            if "Arrhenius" in plot_type:
                synth_ea = st.slider("Activation Energy Ea (kJ/mol):", 30, 150, 75, 5)
                synth_a_factor = st.slider("Frequency Factor ln(A):", 5.0, 25.0, 15.0, 1.0)
            elif "First-Order" in plot_type:
                synth_k = st.slider("Rate Constant k (min⁻¹):", 0.01, 0.10, 0.03, 0.005)
                synth_t_max = st.slider("Time Duration (min):", 20, 120, 60, 10)
            else:
                synth_a0 = st.slider("Initial Concentration [A]₀ (mol/L):", 0.5, 3.0, 1.5, 0.1)
                synth_k_zero = st.slider("Zero-Order Rate Constant k (mol/(L·min)):", 0.01, 0.05, 0.02, 0.005)

        with lab_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            if "Arrhenius" in plot_type:
                slope = -synth_ea * 1000 / 8.314
                xs = np.linspace(0.002, 0.0035, 50)
                ys = slope * xs + synth_a_factor
                ax.plot(xs, ys, color="darkgreen", lw=2)
                ax.set_xlabel("1/T (K⁻¹)", fontsize=11)
                ax.set_ylabel("ln k", fontsize=11)
                ax.set_title(f"Arrhenius Plot (Ea = {synth_ea} kJ/mol, Slope = {slope:.1f})", fontsize=11)
                ax.grid(True, alpha=0.3)
                st.metric("Computed Slope (-Ea/R)", f"{slope:.1f}")
            elif "First-Order" in plot_type:
                xs = np.linspace(0, synth_t_max, 50)
                ys = -synth_k * xs
                ax.plot(xs, ys, color="navy", lw=2)
                ax.set_xlabel("Time, t (min)", fontsize=11)
                ax.set_ylabel("ln([A]/[A]₀)", fontsize=11)
                ax.set_title(f"First-Order Kinetics (k = {synth_k:.4f} min⁻¹)", fontsize=11)
                ax.grid(True, alpha=0.3)
                t_half = 0.693 / synth_k
                st.metric("Computed Half-Life (t½)", f"{t_half:.2f} min")
            else:
                t_end = synth_a0 / synth_k_zero
                xs = np.linspace(0, t_end, 50)
                ys = synth_a0 - synth_k_zero * xs
                ax.plot(xs, ys, color="purple", lw=2)
                ax.set_xlabel("Time, t (min)", fontsize=11)
                ax.set_ylabel("[A] (mol L⁻¹)", fontsize=11)
                ax.set_title(f"Zero-Order Kinetics ([A]₀ = {synth_a0} M)", fontsize=11)
                ax.grid(True, alpha=0.3)
                st.metric("Time to Complete Reaction", f"{t_end:.1f} min")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ==============================================================================
# MODE 3: TRI-MODAL VECTOR DB SEARCH
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
