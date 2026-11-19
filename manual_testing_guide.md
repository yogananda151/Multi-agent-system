# Comprehensive Manual Testing Guide
## Topic-Aware Multi-Agent System for Multimodal JEE Main Chemistry Question Generation & CV Studio

This guide provides a complete, step-by-step walkthrough to manually test, verify, and interact with every component of this project. It covers both **Command-Line Interface (CLI)** testing and the interactive **Fullstack Web Studio (Streamlit Dashboard)**.

---

## 📋 Table of Contents
1. [Prerequisites & System Health Check](#1-prerequisites--system-health-check)
2. [Automated Verification (Sanity Test)](#2-automated-verification-sanity-test)
3. [Manual CLI Testing (Component by Component)](#3-manual-cli-testing-component-by-component)
   - [Step 1: Test Agent 1 (Requirement Analysis)](#step-1-test-agent-1-requirement-analysis)
   - [Step 2: Test Agent 2 (Knowledge Graph & GraphRAG Retrieval)](#step-2-test-agent-2-knowledge-graph--graphrag-retrieval)
   - [Step 3: Test Agent 3 (Examination Prompt Construction)](#step-3-test-agent-3-examination-prompt-construction)
   - [Step 4: Test Agent 4 (Multimodal Question Generation)](#step-4-test-agent-4-multimodal-question-generation)
   - [Step 5: Test Full End-to-End Pipeline](#step-5-test-full-end-to-end-pipeline)
   - [Step 6: Test Packaging & Export](#step-6-test-packaging--export)
4. [Interactive Web Studio Testing (Browser Dashboard)](#4-interactive-web-studio-testing-browser-dashboard)
   - [How to Launch the Dashboard](#how-to-launch-the-dashboard)
   - [Mode 1: 🚀 Multimodal Question Generator](#mode-1--multimodal-question-generator)
   - [Mode 2: 📸 Fullstack Computer Vision Studio](#mode-2--fullstack-computer-vision-studio)
   - [Mode 3: 🔍 Tri-Modal Vector DB Search](#mode-3--tri-modal-vector-db-search)
   - [Mode 4: 🧠 Hybrid GraphRAG Inspector](#mode-4--hybrid-graphrag-inspector)
   - [Mode 5: 📊 System Architecture & Metrics](#mode-5--system-architecture--metrics)
5. [Manual Verification Checklist](#5-manual-verification-checklist)
6. [Troubleshooting & FAQs](#6-troubleshooting--faqs)

---

## 1. Prerequisites & System Health Check

Before beginning testing, open a terminal (PowerShell, Command Prompt, or bash) in the project root directory (`d:\Yoga\Sem_5\DL\D12`).

### 1.1 Verify Python Installation
```powershell
python --version
```
*Expected Output:* `Python 3.10.x` or higher (e.g., Python 3.14.x).

### 1.2 Verify Dependencies
```powershell
pip install -r requirements.txt
```
*Key packages verified:* `torch`, `torchvision`, `qdrant-client`, `fastembed`, `rdkit`, `matplotlib`, `streamlit`, `python-docx`, `networkx`, `pandas`, `numpy`, `pillow`.

### 1.3 Verify Database and Asset Files
Verify that the following essential assets exist in `outputs/`:
```powershell
python -c "
import os
files = [
    'outputs/qdrant_db/meta.json',
    'outputs/chemistry_swin_mlp_features.csv',
    'outputs/chemistry_knowledge_graph.html',
]
for f in files:
    print(f, '->', 'FOUND' if os.path.exists(f) else 'MISSING')
"
```
*Expected Output:* All files reported as `FOUND`.

---

## 2. Automated Verification (Sanity Test)

Run the built-in system verification test suite to confirm that all 9 unit and integration modules are operating nominally:

```powershell
python -m unittest test_system.py
```

### What This Tests:
1. `test_agent1_requirement_analysis`: Natural language parsing of topics, count, difficulty, and format.
2. `test_vector_store_collections`: Qdrant vector database health (794 questions, 2,005 diagram features, 73 molecules).
3. `test_hybrid_graphrag_retrieval`: RRF ($k=60$) rank fusion and 2024–2026 syllabus pruning.
4. `test_agent3_prompt_construction`: Structured exam prompt synthesis.
5. `test_agent4_question_generation`: Programmatic RDKit/Matplotlib generation and multi-format exporters (HTML, DOCX, JSON, SFT JSONL).
6. `test_full_pipeline`: End-to-end multi-agent orchestration.
7. `test_cross_chapter_coupling`: Cross-Chapter coupling for Hard questions (Kinetics $\leftrightarrow$ Equilibrium) and Misconception Diagnostics.
8. `test_multi_format_generation`: Assertion-Reason (A/R) and Section B Numerical integer question formats.
9. `test_vision_suite`: U-Net image restoration, Swin-T 128-D feature extraction, diagram classifier, and reverse CBIR search.

*Expected Output:*
```text
Ran 9 tests in ~15-20s

OK
```

---

## 3. Manual CLI Testing (Component by Component)

### Step 1: Test Agent 1 (Requirement Analysis)
Agent 1 parses unstructured user natural language into structured JSON criteria, validating against the 2024–2026 JEE syllabus.

**Command:**
```powershell
python agent_1_requirement_analysis.py --request "Generate 3 Moderate questions on Chemical Kinetics with diagrams"
```

**Verification Steps:**
1. Check the console output: It should display parsed parameters:
   - Topic: `Chemical Kinetics`
   - Number: `3`
   - Difficulty: `Moderate`
   - Mode: `image`
2. Check the output JSON file at `outputs/requirements.json`:
   ```powershell
   type outputs\requirements.json
   ```
   Confirm that the JSON structure contains keys `Subject`, `Topic`, `Number`, `Difficulty`, `Type`, `Mode`.

---

### Step 2: Test Agent 2 (Knowledge Graph & GraphRAG Retrieval)
Agent 2 retrieves vetted few-shot reference questions and domain concepts using the hybrid combination of NetworkX Knowledge Graph and Qdrant Dense Vector Search with Reciprocal Rank Fusion ($k=60$).

**Command:**
```powershell
python agent_2_kg_retrieval.py
```

**Verification Steps:**
1. Observe the 3-step retrieval pipeline printed to the terminal:
   - `[1/3] Querying Qdrant Vector DB (Dense Embeddings)...`
   - `[2/3] Performing Concept Re-Ranking & Lexical Fusion...`
   - `[3/3] Fusing Dense Vector & Lexical Rankings (RRF)...`
2. Check `outputs/kg_retrieved_context.json`:
   ```powershell
   type outputs\kg_retrieved_context.json
   ```
   Confirm that 3 exemplar questions are returned with their question text, options, answers, and matched concepts.

---

### Step 3: Test Agent 3 (Examination Prompt Construction)
Agent 3 merges the requirements from Agent 1 and retrieved context from Agent 2 into an exam-compliant few-shot generation prompt.

**Command:**
```powershell
python agent_3_prompt_construction.py
```

**Verification Steps:**
1. Inspect the terminal output for:
   `Agent 3: Examination Prompt Construction completed successfully!`
2. Inspect the prompt files:
   - `outputs/question_generation_prompt.txt` (Human-readable text prompt).
   - `outputs/question_generation_prompt.json` (Structured JSON representation).

---

### Step 4: Test Agent 4 (Multimodal Question Generation)
Agent 4 generates scientifically verified chemistry questions with deterministic diagrams (RDKit molecular structures & Matplotlib rate curves), misconception distractor diagnostics, and multi-format exporters.

#### Test 4A: Standard Multiple-Choice Questions (MCQ)
```powershell
python agent_4_question_generation.py --num 2 --topic "Chemical Kinetics" --format MCQ
```

#### Test 4B: Assertion-Reason Questions (CD-NS-KG-MAS Paper Format)
```powershell
python agent_4_question_generation.py --num 1 --topic "Chemical Kinetics" --format "Assertion-Reason"
```

#### Test 4C: Section B Numerical / Integer Questions
```powershell
python agent_4_question_generation.py --num 1 --topic "Chemical Kinetics" --format Numerical
```

**Verification Steps:**
1. **Diagrams**: Open `outputs/generated_question_images/` and view the freshly generated `.png` files (e.g. `graph_001.png`, `graph_002.png`). Confirm clear axes, readable fonts, and chemical annotations.
2. **Interactive HTML**: Double-click `outputs/generated_chemistry_questions.html` to open in your browser. Verify styling, embedded diagrams, correct answer badges, and explanations.
3. **Word Document**: Open `outputs/generated_chemistry_questions.docx` in Microsoft Word or LibreOffice. Verify header, questions, inline diagrams, and green correct answer keys.
4. **Fine-Tuning Dataset**: Inspect `outputs/sft_synthetic.jsonl` to verify instruction-tuning records conforming to the `prompt`/`completion` or `instruction`/`input`/`output` schema.

---

### Step 5: Test Full End-to-End Pipeline
Test the complete automated coordinator linking Agent 1 $\to$ Agent 2 $\to$ Agent 3 $\to$ Agent 4.

**Command:**
```powershell
python pipeline.py --request "Generate 2 Hard questions on Chemical Kinetics"
```

**Verification Steps:**
1. Verify the console logs:
   - Step 1 parses the request.
   - For `Hard` difficulty, verify the **Cross-Chapter Coupling** alert:
     `[CROSS-CHAPTER COUPLING] Active for Hard Question Setting:`
     `Coupling: Chemical Kinetics <---> Equilibrium`
     `Formula : K_eq = k_forward / k_backward; d(ln K)/dT = Delta H / (R * T^2)`
   - Step 2 runs GraphRAG retrieval with reciprocal rank fusion.
   - Step 3 builds the prompt.
   - Step 4 synthesizes the questions and exports JSON, HTML, and DOCX files.
2. Confirm the exit status is `0` with the final report banner:
   `======================================================================`
   `CHEMISTRY PIPELINE EXECUTION COMPLETED SUCCESSFULLY`
   `======================================================================`

---

### Step 6: Test Packaging & Export
Package all generated outputs into a distribution-ready `.zip` archive.

**Command:**
```powershell
python export_outputs.py
```

**Verification Steps:**
1. Verify that `outputs/chemistry_outputs.zip` is created.
2. Verify its file size (~50–60 MB, including diagram image assets).

---

## 4. Interactive Web Studio Testing (Browser Dashboard)

The project includes a production-grade Streamlit web application integrating all multi-agent capabilities, the Computer Vision studio, vector database inspection, and knowledge graph visualization.

### How to Launch the Dashboard

Run the following command in your terminal:
```powershell
python -m streamlit run app.py
```
*(Alternatively, on Windows you can double-click [`run_dashboard.bat`](file:///d:/Yoga/Sem_5/DL/D12/run_dashboard.bat)).*

Open your web browser and navigate to:
**`http://localhost:8501`**

---

### Mode 1: 🚀 Multimodal Question Generator
*Test question generation directly from the visual interface.*

1. In the left sidebar under **"Choose Studio Mode"**, select **"🚀 Multimodal Question Generator"**.
2. Set the configuration parameters:
   - **Target Topic:** Select `Chemical Kinetics` (or `Coordination Compounds`, `Solutions`, etc.).
   - **Question Format:** Choose `Standard 4-Option MCQ`, `Assertion-Reason (Type A/R)`, or `Section B Integer/Numerical`.
   - **Difficulty Level:** Select `Hard` (notice the dynamic warning pill showing active cross-chapter coupling with Equilibrium).
   - **Number of Questions:** Set slider to `2`.
3. Click the primary button: **"⚡ Generate Verified Multimodal Questions"**.
4. **What to Verify:**
   - Real-time progress bar advances through Agent 1 $\to$ Agent 2 $\to$ Agent 3 $\to$ Agent 4.
   - High-resolution question cards render with:
     - Question text and math formulas.
     - Embedded Matplotlib / RDKit diagram.
     - Clickable options ($A, B, C, D$).
     - Green correct answer highlight.
     - **CD-NS-KG-MAS Misconception Diagnostics** breakdown for distractors.
   - Scroll to the bottom and test download buttons:
     - `📥 Download Interactive HTML (.html)`
     - `📥 Download Microsoft Word (.docx)`
     - `📥 Download Machine JSON (.json)`
     - `📥 Download SFT Training Dataset (.jsonl)`
     - `📦 Download Complete Packaged Bundle (.zip)`

---

### Mode 2: 📸 Fullstack Computer Vision Studio
*Test the 5 deep learning visual studio tabs (derived from `Unet+Swin.ipynb`).*

In the sidebar, select **"📸 Fullstack Computer Vision Studio"**.

#### Tab 1: 🔬 Neural Diagram Restoration (PyTorch U-Net)
1. Under **"Select Image Source"**, choose **"Choose Sample Diagram from Project"**.
2. Pick any sample diagram (e.g. `graph_001.png` or `inorg_001.png`).
3. Set the enhancement sliders:
   - **Simulate Gaussian Noise:** `0.05`
   - **Restoration Sharpness:** `1.50`
   - **Contrast Multiplier:** `1.20`
4. Click **"Run U-Net Diagram Restoration"**.
5. **What to Verify:**
   - A side-by-side comparison appears: **Original / Noisy Diagram** vs **Restored & Denoised Diagram**.
   - Chemical bonds, text, and axes are sharp and clean.
   - Click the **"📥 Download Restored PNG"** button and verify the saved image.

#### Tab 2: 🧬 Swin-T Visual Features & Diagram Anatomy
1. Select a diagram from the dropdown.
2. **What to Verify:**
   - **Swin-T Latent Embedding Heatmap**: Displays the 128-D normalized visual representation vector extracted via the Swin-T + MLP backbone.
   - **Diagram Anatomy Metrics**: Four progress bars indicating:
     - *Foreground Ratio* (% of diagram pixels)
     - *Contrast Ratio* (dynamic range)
     - *Edge Density* (Sobel filter bond & boundary score)
     - *Spatial Complexity* (entropy of high-frequency visual components)

#### Tab 3: 🏷️ Visual Diagram Classifier
1. Select any sample diagram.
2. Click **"Classify Diagram Structure"**.
3. **What to Verify:**
   - The classifier identifies the category (e.g. `Kinetics Reaction Coordinate & Decay Curve`, `Organic / Molecular Skeletal Structure`, `Inorganic / Electrochemical / Apparatus`).
   - A horizontal confidence score bar shows the class probabilities.

#### Tab 4: 🔎 Reverse Diagram Search (Qdrant CBIR)
1. Select a query diagram.
2. Set **"Top Matches to Retrieve"** to `3`.
3. Click **"Execute Reverse Diagram Search"**.
4. **What to Verify:**
   - The engine searches across 2,005 indexed diagram vectors in Qdrant.
   - Returns the Top 3 most visually and conceptually similar diagrams with similarity scores (e.g. `Cosine Similarity: 94.2%`).

#### Tab 5: 🎨 Synthetic Diagram Laboratory
1. Select a diagram preset:
   - `Arrhenius Plot (ln k vs 1/T)`
   - `First-Order Reaction Decay (ln[A] vs Time)`
   - `Zero-Order Reaction Decay ([A] vs Time)`
   - `Octahedral Crystal Field Splitting (d-Orbitals)`
2. Tweak dynamic parameters (e.g. Activation Energy $E_a$, Temperature range, Rate constant $k$).
3. Click **"Synthesize & Export Vector Diagram"**.
4. **What to Verify:**
   - A fresh, high-resolution vector diagram is mathematically generated on the fly.
   - Computed metrics display (e.g. Calculated Slope $-E_a/R$, half-life $t_{1/2}$).
   - Download button allows saving the generated plot.

---

### Mode 3: 🔍 Tri-Modal Vector DB Search
*Test semantic and chemical similarity search directly.*

In the sidebar, select **"🔍 Tri-Modal Vector DB Search"**.

1. **Sub-Tab 1: Semantic Question Search**:
   - Enter a query like: `"half life of a first order reaction"`
   - Click **"Search Questions"**.
   - Verify that semantically related JEE chemistry questions are returned with similarity scores.
2. **Sub-Tab 2: Molecular Fingerprint Search (RDKit Morgan ECFP4)**:
   - Enter a SMILES string (e.g. `CC(=O)Oc1ccccc1C(=O)O` for Aspirin).
   - Click **"Search Similar Molecules"**.
   - Verify that structurally similar compounds are retrieved from the 73 indexed molecules with Tanimoto coefficient ranking.

---

### Mode 4: 🧠 Hybrid GraphRAG Inspector
*Inspect the symbolic knowledge structure and syllabus alignment.*

In the sidebar, select **"🧠 Hybrid GraphRAG Inspector"**.

1. **Syllabus Pruning Tab**: Verify the list of officially pruned 2024–2026 JEE Main chapters (Solid State, Surface Chemistry, States of Matter, Metallurgy, Hydrogen, s-Block, Polymers, Everyday Chemistry).
2. **Cross-Chapter Coupling Tab**: Inspect the 5 hard-difficulty coupling matrices (Kinetics $\leftrightarrow$ Equilibrium, Thermodynamics $\leftrightarrow$ Electrochemistry, etc.).
3. **Interactive Graph Visualizer**: Open the embedded Pyvis interactive network graph or click the link to view [chemistry_knowledge_graph.html](file:///d:/Yoga/Sem_5/DL/D12/outputs/chemistry_knowledge_graph.html).

---

### Mode 5: 📊 System Architecture & Metrics
*Inspect runtime metrics and hardware acceleration.*

In the sidebar, select **"📊 System Architecture & Metrics"**.

1. Verify the Mermaid flow architecture diagram.
2. Check Qdrant collection statistics:
   - `chemistry_questions`: 794 points
   - `diagram_embeddings`: 2,005 points
   - `molecular_structures`: 73 points
3. Check **Hardware & Device Runtime**:
   - Device: `cuda` (if NVIDIA GPU present) or `cpu`.
   - 4-Bit Quantization support status.
   - Python version & platform architecture.

---

## 5. Manual Verification Checklist

Use this checklist during testing to record results:

| # | Test Area | Test Action | Expected Result | Pass/Fail |
|---|---|---|---|---|
| 1 | **Unit Test Suite** | Run `python -m unittest test_system.py` | 9/9 tests pass with `OK` | [ ] |
| 2 | **Agent 1** | Run `python agent_1_requirement_analysis.py --request "..."` | `outputs/requirements.json` generated | [ ] |
| 3 | **Agent 2** | Run `python agent_2_kg_retrieval.py` | `outputs/kg_retrieved_context.json` created with 3 exemplars | [ ] |
| 4 | **Agent 3** | Run `python agent_3_prompt_construction.py` | `outputs/question_generation_prompt.txt` created | [ ] |
| 5 | **Agent 4 MCQ** | Run `python agent_4_question_generation.py --format MCQ` | Generates 4-option MCQ with diagram | [ ] |
| 6 | **Agent 4 A/R** | Run with `--format "Assertion-Reason"` | Generates Assertion/Reason pair with standard options | [ ] |
| 7 | **Agent 4 Num** | Run with `--format Numerical` | Generates Section B integer question | [ ] |
| 8 | **Master Pipeline** | Run `python pipeline.py --request "..."` | Full 4-agent run completes successfully | [ ] |
| 9 | **Hard Coupling** | Run pipeline with difficulty `Hard` | Cross-chapter coupling alert triggered | [ ] |
| 10 | **Packaging** | Run `python export_outputs.py` | `chemistry_outputs.zip` created | [ ] |
| 11 | **Dashboard Launch**| Run `python -m streamlit run app.py` | Server starts on port 8501 without errors | [ ] |
| 12 | **U-Net Restoration**| In CV Studio Tab 1, run diagram restoration | Before/after comparison displayed | [ ] |
| 13 | **Swin-T Latent** | In CV Studio Tab 2, view heatmap | 128-D vector heatmap & metrics rendered | [ ] |
| 14 | **Diagram Classifier**| In CV Studio Tab 3, classify diagram | Class probabilities & prediction displayed | [ ] |
| 15 | **Reverse Search** | In CV Studio Tab 4, run CBIR search | Top-K similar diagrams retrieved from Qdrant | [ ] |
| 16 | **Diagram Lab** | In CV Studio Tab 5, synthesize Arrhenius plot | High-resolution diagram generated | [ ] |
| 17 | **Vector DB Search**| In Mode 3, search `"half life"` | Semantic question matches returned | [ ] |
| 18 | **Export Buttons** | Click HTML, DOCX, JSON, ZIP download buttons | All files download cleanly with valid data | [ ] |

---

## 6. Troubleshooting & FAQs

### Q1: `streamlit: command not found` or `The term 'streamlit' is not recognized` (or blocked by Windows Defender/WDAC)
**Resolution:** Run Streamlit via the Python module launcher:
```powershell
python -m streamlit run app.py
```
Or double-click [`run_dashboard.bat`](file:///d:/Yoga/Sem_5/DL/D12/run_dashboard.bat).

### Q2: Port 8501 is already in use
**Resolution:** Launch Streamlit on an alternate port:
```powershell
python -m streamlit run app.py --server.port 8502
```

### Q3: `outputs/qdrant_db` locked or SQLite busy
**Resolution:** Ensure no other Python process is holding the database open, or delete any stale `.lock` file:
```powershell
python -c "import os; os.remove('outputs/qdrant_db/.lock') if os.path.exists('outputs/qdrant_db/.lock') else None"
```

### Q4: How do I re-populate the vector database if I delete it?
**Resolution:** Run the vector database population script:
```powershell
python populate_vector_db.py
```
This re-indexes all 794 chemistry questions, 2,005 diagram features, and 73 molecular structures into local Qdrant collections.
