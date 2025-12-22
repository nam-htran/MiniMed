# Tệp: main.py
import logging
import sys
import time
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

# --- 1. ĐỊNH NGHĨA BỘ LỌC RÁC (Custom Filter) ---
class AntiNoiseFilter(logging.Filter):
    """Bộ lọc đặc biệt để chặn các log rác cứng đầu từ thư viện bên thứ 3."""
    def filter(self, record):
        msg = record.getMessage()
        if "eligible syntax" in msg:
            return False
        if "Loading faiss" in msg:
            return False
        return True

# --- 2. CẤU HÌNH LOGGING ---
NOISY_LIBS = [
    "PyRuSH", "presidio-analyzer", "medspacy", "urllib3", 
    "sentence_transformers", "httpx", "httpcore", "hpack", 
    "google.ai", "google.auth", "neo4j", "huggingface_hub", 
    "transformers", "faiss.loader", "faiss", "gliner", 
    "pdfminer", "charset_normalizer", "google_genai.models"
]
for lib in NOISY_LIBS:
    logging.getLogger(lib).setLevel(logging.CRITICAL)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S', 
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(AntiNoiseFilter())

# --- 3. IMPORT CÁC MODULE CỦA PIPELINE ---
from src.core.state import MedCOTState
from src.modules import (
    step0_preprocess, step1_extraction, step2_linking, 
    step4_retrieval, step5_reasoning, step6_path_generation, 
    step7_verification, step8_synthesis, step9_safety, step10_logging
)
from src.utils.neo4j_connect import db_connector

logger = logging.getLogger("MED-COT_MAIN")

# --- 4. HÀM CHẠY PIPELINE CHÍNH ---
def run_pipeline(query: str, patient_context: str = None, config: dict = None):
    if not db_connector:
        logger.critical("❌ Kết nối Neo4j thất bại. Dừng pipeline.")
        return None
        
    cfg = config or {}
    use_gcot = cfg.get("use_gcot", True)
    
    logger.info(f"{'='*50}\n🚀 RUNNING PIPELINE (FINAL CLEAN)\n🚀 QUERY: '{query}'\n{'='*50}")
    state = MedCOTState(raw_query=query, patient_context=patient_context)
    start_time = time.time()
    
    try:
        logger.info("\n--- 🏁 PHASE 1: DATA PREPARATION ---")
        state = step0_preprocess.run(state)
        state = step1_extraction.run(state)
        state = step2_linking.run(state)
        
        logger.info("\n--- ⚡ PHASE 2: REASONING & RETRIEVAL ---")
        state = step4_retrieval.run(state)
        
        if use_gcot:
            state = step5_reasoning.run(state)
            
        state = step6_path_generation.run(state)
        state = step7_verification.run(state)
        
        # --- SỬA ĐỔI THỨ TỰ THỰC THI ---
        logger.info("\n--- 🔬 PHASE 3: SYNTHESIS & SAFETY ---")
        # Chạy safety check lần 1 để tạo `safety_flags` cho prompt của LLM
        state = step9_safety.run(state)
        
        # Tổng hợp câu trả lời dựa trên tất cả bằng chứng, bao gồm cả safety_flags
        state = step8_synthesis.run(state)
        
        # Chạy safety check lần 2 để đảm bảo khối cảnh báo được chèn vào đầu câu trả lời cuối cùng
        state = step9_safety.run(state)
        # -------------------------------

        logger.info("\n--- 📝 PHASE 4: LOGGING ---")
        step10_logging.run(state)

    except Exception as e:
        logger.exception(f"Critical pipeline error: {e}")
    finally:
        total_time = time.time() - start_time
        logger.info(f"\n{'='*50}\n🏁 PIPELINE FINISHED IN {total_time:.2f} SECONDS\n{'='*50}")
    
    return state

# --- 5. HÀM HIỂN THỊ KẾT QUẢ ---
def inspect_and_display(state: MedCOTState):
    """In kết quả cuối cùng ra màn hình console một cách đẹp mắt."""
    print(f"\n\033[1m\033[94m--- FINAL RESULT ---\033[0m\n")
    print(f"❓ Query: {state.raw_query}\n")
    print(f"💡 ANSWER:\n{state.final_answer or 'No answer generated.'}\n")
    if state.safety_flags:
        print(f"\033[91m🚨 Safety Flags Detected: {len(state.safety_flags)}\033[0m")
        for flag in state.safety_flags:
            print(f"  - {flag['msg']}")
    print("\n" + "="*50)

# ==============================================================================
#  6. ĐIỂM KHỞI CHẠY CHÍNH (ENTRY POINT)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full MedCOT Neuro-Symbolic Pipeline.")
    parser.add_argument("--query", type=str, required=True, help="The medical question to analyze.")
    parser.add_argument("--context", type=str, default=None, help="(Optional) Patient-specific context.")
    parser.add_argument("--no-gcot", action="store_true", help="(Optional) Disable the GNN reasoning step (Step 5).")
    args = parser.parse_args()
    
    final_state = run_pipeline(query=args.query, patient_context=args.context, config={"use_gcot": not args.no_gcot})
    
    if final_state:
        inspect_and_display(final_state)
        
    if db_connector:
        db_connector.close()