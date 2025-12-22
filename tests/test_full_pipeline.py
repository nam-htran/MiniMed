# tests/test_full_pipeline.py
import logging
import sys
import time
import os

# --- CRITICAL: CONFIGURE LOGGING FIRST ---
# Phải đặt trước tất cả các import khác để có hiệu lực
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True  # Ghi đè mọi cấu hình logging đã có
)

# Tắt tiếng ồn từ các thư viện cụ thể bằng cách set level CRITICAL
for lib in ["PyRuSH", "presidio-analyzer", "medspacy", "urllib3", "sentence_transformers", "httpx", "httpcore", "hpack", "google.ai"]:
    logging.getLogger(lib).setLevel(logging.CRITICAL)
# ------------------------------------------

from src.core.state import MedCOTState
from src.modules import (
    step0_preprocess, step1_extraction, step2_linking,
    step4_retrieval, step5_reasoning, step6_path_generation,
    step7_verification, step8_synthesis, step9_safety, step10_logging
)
from src.utils.neo4j_connect import db_connector

logger = logging.getLogger("TEST_PIPELINE")

def print_section(title):
    print(f"\n{'='*60}\n🚀 {title}\n{'='*60}")

def inspect_advanced_features(state: MedCOTState):
    print(f"\n🔍 --- KIỂM TRA TÍNH NĂNG NÂNG CAO ---")
    linked = [e for e in state.linked_entities if e.link_status == "linked"]
    if linked: 
        names = [c.best_candidate.preferred_name for c in linked[:3]]
        print(f"✅ [Step 2] Linked: {names}")
    
    graph = state.graph_refs.get("ckg_subgraph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    print(f"✅ [Step 4] Subgraph: {len(nodes)} nodes, {len(edges)} edges")
    
    print(f"✅ [Step 6] Candidate Paths Found: {len(state.candidate_paths)}")
    
    if state.verified_path:
        print(f"✅ [Step 7] Verification Confidence: {state.global_confidence:.4f}")
    else:
        print("⚠️ [Step 7] No path passed verification threshold.")
        
    print(f"✅ [Step 9] Safety Flags Triggered: {len(state.safety_flags)}")

def run_test_case(query, context=None):
    if db_connector is None: 
        logger.critical("Database connector not available. Aborting.")
        return
        
    start_t = time.time()
    state = MedCOTState(raw_query=query, patient_context=context)

    try:
        state = step0_preprocess.run(state)
        state = step1_extraction.run(state)
        state = step2_linking.run(state)
        state = step4_retrieval.run(state)
        state = step5_reasoning.run(state)
        state = step6_path_generation.run(state)
        state = step7_verification.run(state)
        state = step8_synthesis.run(state)
        state = step9_safety.run(state)
        state = step10_logging.run(state)
        
        print_section("KẾT QUẢ CUỐI CÙNG")
        print(f"❓ Query: {query}")
        print(f"\n💡 ANSWER:\n{state.final_answer}\n")
        
        inspect_advanced_features(state)
        
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
    
    print(f"\n⏱️ Total Time: {time.time() - start_t:.2f}s")

if __name__ == "__main__":
    print_section("TEST CASE 1: SIMPLE QUERY")
    run_test_case("What are the treatments for hypertension?")

    print_section("TEST CASE 2: SAFETY & DDI CHECK")
    run_test_case("Is it safe to take Warfarin and Aspirin together?")

