import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.core.state import MedCOTState
from src.modules import step0_preprocess, step1_extraction, step2_linking, step4_retrieval
from src.modules.step5_reasoning import _prepare_hetero_data_robust, load_encoder
from src.utils.neo4j_connect import db_connector

DATA_FILE = "data/medcot_rich_training_data.jsonl"
OUTPUT_DIR = Path("data/processed_gnn_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not db_connector: return
    encoder = load_encoder()
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu đầu vào {DATA_FILE}")
        return
        
    df = pd.read_json(DATA_FILE, lines=True)
    
    # --- SỬA LOGIC LỌC ĐỂ CHẶT CHẼ HƠN ---
    if 'verified_path_text' not in df.columns:
        print(f"❌ Lỗi: Không tìm thấy cột 'verified_path_text' trong {DATA_FILE}.")
        return

    # Lọc các dòng có verified_path_text không rỗng và là một chuỗi
    df_filtered = df[df['verified_path_text'].apply(lambda x: isinstance(x, str) and len(x) > 5)].copy()
    
    if df_filtered.empty:
        print(f"⚠️ Không có mẫu hợp lệ nào trong {DATA_FILE} để tạo dữ liệu GNN.")
        return
    # ----------------------------------------
    
    print(f"Processing {len(df_filtered)} samples for GNN dataset...")
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            state = MedCOTState(raw_query=row['question'])
            state = step0_preprocess.run(state)
            state = step1_extraction.run(state)
            state = step2_linking.run(state)
            state = step4_retrieval.run(state, top_k_nodes=100)
            
            ug = state.graph_refs.get("ckg_subgraph")
            if not ug or not ug.get("nodes"): continue

            ckg_n = [n for n in ug["nodes"] if n.get("label") not in ["Patient", "Observation"]]
            psg_n = [n for n in ug["nodes"] if n.get("label") in ["Patient", "Observation"]]
            ckg_e = [e for e in ug["edges"] if e.get("provenance") != "PSG"]
            psg_e = [e for e in ug["edges"] if e.get("provenance") == "PSG"]

            ckg_d, _ = _prepare_hetero_data_robust(ckg_n, ckg_e, encoder)
            psg_d, _ = _prepare_hetero_data_robust(psg_n, psg_e, encoder)
            
            if not ckg_d.edge_types: continue

            # Dùng index gốc để đặt tên file cho nhất quán
            original_index = row.name 
            torch.save({
                "ckg_data": ckg_d,
                "psg_data": psg_d,
                "query_text": state.normalized_query,
                "target_path": row['verified_path_text']
            }, OUTPUT_DIR / f"sample_{original_index}.pt")
            
        except Exception as e:
            # print(f"Skipping sample {idx} due to error: {e}")
            pass

if __name__ == "__main__":
    main()