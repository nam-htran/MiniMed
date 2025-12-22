# run/build_faiss_index.py
import json
import logging
import numpy as np
import faiss
import os
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.utils.neo4j_connect import db_connector

# --- CONFIG ---
MODEL_NAME = "BAAI/bge-small-en-v1.5" # Model nhỏ, nhanh, hiệu quả
OUTPUT_DIR = Path("data/kg_index")
INDEX_PATH = OUTPUT_DIR / "kg_faiss.index"
META_PATH = OUTPUT_DIR / "kg_nodes_meta.json"
BATCH_SIZE = 5000  # Xử lý 5000 node mỗi lần để tiết kiệm RAM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FAISS_BUILDER")

def main():
    # 1. Kiểm tra nếu Index đã tồn tại thì Skip
    if INDEX_PATH.exists() and META_PATH.exists():
        print(f"\n⏩ [SKIP] FAISS Index đã tồn tại tại: {OUTPUT_DIR}")
        print("👉 Nếu bạn vừa nạp dữ liệu mới và muốn build lại, hãy xóa thư mục 'data/kg_index' rồi chạy lại script này.")
        return

    # 2. Kiểm tra kết nối DB
    if db_connector is None:
        logger.error("❌ Không có kết nối Neo4j. Vui lòng kiểm tra Docker.")
        return

    # Tạo thư mục output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Đếm tổng số node để hiển thị thanh tiến trình
    logger.info("📊 Đang đếm tổng số node cần index...")
    count_query = "MATCH (n) WHERE n.name IS NOT NULL RETURN count(n) as total"
    try:
        res = db_connector.run_query(count_query)
        total_nodes = res[0]['total']
        logger.info(f"   -> Tổng số node: {total_nodes}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi đếm node: {e}")
        return

    # 4. Khởi tạo Model & Index
    logger.info(f"🧠 Loading SentenceTransformer: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)
    
    # Sử dụng IndexFlatIP (Inner Product) cho cosine similarity (khi vectors đã normalize)
    # Loại này tiết kiệm RAM hơn HNSW và vẫn đủ nhanh cho vài triệu node.
    embedding_dim = 384
    index = faiss.IndexFlatIP(embedding_dim) 

    all_meta = []
    
    # 5. Vòng lặp Batch Processing (Tiết kiệm RAM)
    logger.info("🚀 Bắt đầu quá trình Indexing theo batch...")
    
    query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
    RETURN elementId(n) AS node_id, labels(n) AS labels, n.name AS name
    ORDER BY elementId(n)
    SKIP $skip LIMIT $limit
    """
    
    skip = 0
    pbar = tqdm(total=total_nodes, desc="Indexing Nodes", unit="node")

    while skip < total_nodes:
        # A. Fetch Batch từ Neo4j
        rows = db_connector.run_query(query, {"skip": skip, "limit": BATCH_SIZE})
        if not rows:
            break
            
        batch_meta = []
        batch_texts = []
        
        # B. Prepare Data
        for r in rows:
            # Xử lý an toàn dữ liệu
            lbls = r.get("labels", [])
            lbl = lbls[0] if lbls else "Unknown"
            name = r.get("name", "Unknown")
            nid = str(r.get("node_id"))
            
            # Lưu metadata gọn nhẹ
            meta_item = {
                "node_id": nid,
                "labels": lbls,
                "name": name
            }
            batch_meta.append(meta_item)
            
            # Text để embed: "Name (Label)"
            batch_texts.append(f"{name} ({lbl})")
        
        # C. Encode Batch (GPU/CPU)
        if batch_texts:
            embeddings = encoder.encode(
                batch_texts,
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=True # Quan trọng cho FlatIP/Cosine
            )
            
            # D. Add to FAISS Index
            index.add(np.asarray(embeddings, dtype="float32"))
            
            # E. Append Meta
            all_meta.extend(batch_meta)
        
        skip += BATCH_SIZE
        pbar.update(len(rows))

    pbar.close()

    # 6. Lưu xuống đĩa
    logger.info(f"💾 Đang lưu FAISS index vào {INDEX_PATH}...")
    faiss.write_index(index, str(INDEX_PATH))

    logger.info(f"💾 Đang lưu Metadata vào {META_PATH}...")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=None) # indent=None cho file nhỏ gọn

    logger.info("🎉 Hoàn tất build FAISS index!")
    if db_connector:
        db_connector.close()

if __name__ == "__main__":
    main()