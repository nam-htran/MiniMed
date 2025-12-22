import os
import json
import logging
import pandas as pd
import gc  # Garbage collection để quản lý RAM
from tqdm import tqdm
from src.core.state import MedCOTState
from src.modules import (
    step0_preprocess, step1_extraction, step2_linking,
    step4_retrieval, step5_reasoning, step6_path_generation,
    step7_verification
)
from src.utils.neo4j_connect import db_connector
# Import module Local LLM mới tạo
from src.utils.local_llm import local_llm

# --- CONFIG ---
logging.basicConfig(level=logging.ERROR)
INPUT_PARQUET = "data/medical_o1_vi_translated_EVALUATED_GEMINI.parquet"
OUTPUT_JSONL = "data/medcot_rich_training_data.jsonl"

def generate_raw_trace(query: str):
    """
    Chạy pipeline MedCOT (Step 0 -> Step 7) để lấy dữ liệu thô từ Knowledge Graph.
    """
    state = MedCOTState(raw_query=query)
    try:
        # Chạy các bước logic đồ thị
        state = step0_preprocess.run(state)
        state = step1_extraction.run(state)
        state = step2_linking.run(state)
        state = step4_retrieval.run(state)
        state = step5_reasoning.run(state)
        state = step6_path_generation.run(state)
        state = step7_verification.run(state)
        
        path_text = ""
        # state.gcot['verified_path_text'] được gán trong step7
        if state.gcot.get('verified_path_text'):
            path_text = state.gcot['verified_path_text']
        elif state.candidate_paths:
            path_text = state.candidate_paths[0].get('text_repr', "")

        raw_info = {
            "query": query,
            "seed_nodes": [e.best_candidate.preferred_name for e in state.linked_entities if e.link_status == 'linked'],
            "graph_context": state.gcot.get("graph_tokens", ""),
            "verified_path_text": path_text, # Tên cột quan trọng
            "confidence": state.global_confidence
        }
        return raw_info
    except Exception as e:
        return None

def normalize_cot_with_llm(raw_info):
    """
    Sử dụng Local LLM để viết lại suy luận.
    """
    if not raw_info or not raw_info["verified_path_text"]:
        return "Reasoning could not be generated due to lack of graph evidence."

    prompt = f"""
    Analyze the medical Knowledge Graph path below and explain the reasoning step-by-step to answer the question.
    
    **Question:** "{raw_info['query']}"
    **Entities:** {', '.join(raw_info['seed_nodes'])}
    **Graph Path:** "{raw_info['verified_path_text']}"
    
    **Requirement:** Write a concise, logical Chain-of-Thought based on this path.
    """
    
    try:
        return local_llm.generate_cot(prompt)
    except Exception as e:
        return f"Local LLM Error: {e}"

def main():
    if db_connector is None:
        print("❌ Neo4j connection failed. Please check docker container.")
        return
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"❌ Input file not found: {INPUT_PARQUET}")
        return

    print(f"🚀 Loading data from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # Chạy test với 10 dòng đầu
    # df = df.head(10) 
    
    print("⏳ Warming up Local LLM (DeepSeek-R1-1.5B)...")
    try:
        local_llm.load_model()
    except Exception as e:
        print(f"❌ Failed to load Local LLM: {e}")
        return

    results = []
    
    print("running...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Rich Traces"):
        raw_trace_info = generate_raw_trace(row['Question'])
        
        if raw_trace_info:
            normalized_medcot = normalize_cot_with_llm(raw_trace_info)
        else:
            normalized_medcot = "Pipeline failed to generate trace."

        # --- SỬA ĐỔI Ở ĐÂY ---
        # Thêm cột 'verified_path_text' vào dictionary kết quả để lưu xuống file
        results.append({
            "question": row['Question'],
            "answer": row['Response'],
            "default_cot": row['Complex_CoT'],
            "medcot_cot": normalized_medcot,
            "verified_path_text": raw_trace_info.get("verified_path_text", "") if raw_trace_info else ""
        })
        # ---------------------
        
        if idx % 10 == 0:
            gc.collect()

    print(f"💾 Saving {len(results)} rows to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    local_llm.unload()
    
    if db_connector: 
        db_connector.close()
        
    print(f"✅ DONE! Rich CoT data ready at: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()