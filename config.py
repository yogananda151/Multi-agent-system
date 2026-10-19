"""
Configuration Module for Chemistry Multi-Agent System
Handles directories, file paths, model configurations, and device settings.
Supports Windows, Linux, Colab, and Kaggle environments seamlessly.
"""

import os
import gc
import sys

# ==============================================================================
# ENVIRONMENT & DIRECTORY RESOLUTION
# ==============================================================================

# Determine base working directory
if os.path.exists("/kaggle/working"):
    BASE_DIR = "/kaggle/working"
elif "COLAB_GPU" in os.environ or os.path.exists("/content"):
    BASE_DIR = "/content"
else:
    # Local workspace
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Primary output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs") if BASE_DIR != "/kaggle/working" else "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image directories
IMAGE_ROOT = os.path.join(OUTPUT_DIR, "chemistry_images")
GENERATED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "generated_question_images")
os.makedirs(IMAGE_ROOT, exist_ok=True)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
DATASET_NAME = "eQOURSE/jee-main-questions"
DATASET_CONFIG = "chemistry"

# Preprocessed CSV file
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "chemistry_text_questions_preprocessed.csv")

# ==============================================================================
# KNOWLEDGE GRAPH PATHS
# ==============================================================================
KG_PICKLE_PATH = os.path.join(OUTPUT_DIR, "chemistry_knowledge_graph.pkl")
KG_HTML_PATH = os.path.join(OUTPUT_DIR, "chemistry_knowledge_graph.html")
REBUILD_KG_IF_STALE = True

# ==============================================================================
# AGENT INTER-COMMUNICATION PATHS
# ==============================================================================
# Agent 1 outputs
AGENT1_REQUIREMENTS_PATH = os.path.join(OUTPUT_DIR, "requirements.json")

# Agent 2 outputs
AGENT2_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "kg_retrieved_context.json")
AGENT2_OUTPUT_PKL = os.path.join(OUTPUT_DIR, "kg_retrieved_context.pkl")

# Agent 3 outputs
AGENT3_OUTPUT_TXT = os.path.join(OUTPUT_DIR, "question_generation_prompt.txt")
AGENT3_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "question_generation_prompt.json")

# Agent 4 outputs
AGENT4_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "generated_chemistry_questions.json")
AGENT4_OUTPUT_PKL = os.path.join(OUTPUT_DIR, "generated_chemistry_questions.pkl")
AGENT4_OUTPUT_HTML = os.path.join(OUTPUT_DIR, "generated_chemistry_questions.html")
AGENT4_OUTPUT_DOCX = os.path.join(OUTPUT_DIR, "generated_chemistry_questions.docx")

# Zipped archive output
ZIP_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "chemistry_outputs")

# ==============================================================================
# MODEL & GENERATION HYPERPARAMETERS
# ==============================================================================
MODEL_NAME = os.environ.get("CHEM_AGENT_MODEL", "Qwen/Qwen3-8B")

BATCH_SIZE = 5
MAX_BATCH_ATTEMPTS = 8
MAX_NEW_TOKENS = 2200
TEMPERATURE = 0.70
RETRY_TEMPERATURE = 0.55
TOP_P = 0.90
TOP_K = 40
REPETITION_PENALTY = 1.05
MAX_AGENT3_CONTEXT_TOKENS = 1500

# Verification settings
VERIFY_TEXT = True
VERIFY_MAX_TOKENS = 600
STRICT_VERIFY_ATTEMPTS = 5
DUP_SEQ_RATIO = 0.75
DUP_JACCARD = 0.60

# ==============================================================================
# DEVICE & ACCELERATION HELPERS
# ==============================================================================
def get_device_info():
    """Returns detected device, 4-bit quantization support flag, and description."""
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import bitsandbytes
                has_bnb = True
            except ImportError:
                has_bnb = False
            return {
                "device": "cuda",
                "use_4bit": has_bnb,
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda
            }
    except ImportError:
        pass
    
    return {
        "device": "cpu",
        "use_4bit": False,
        "gpu_name": None,
        "cuda_version": None
    }


def clear_cuda_memory():
    """Safely triggers garbage collection and flushes CUDA cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def show_gpu_memory(label=""):
    """Displays current GPU memory allocation if CUDA is active."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory {label}: Allocated={allocated:.2f} GB | "
              f"Reserved={reserved:.2f} GB | Total={total:.2f} GB")
    except Exception:
        pass
