# run/5_ingest_custom_data.py
import os
import json
import logging
import glob
import re
from pathlib import Path
from src.utils.neo4j_connect import db_connector
from src.utils.local_llm import local_llm

# --- CẤU HÌNH ---
DATA_DIR = Path("data/custom_knowledge")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging ra màn hình
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)

logger = logging.getLogger("KG_BUILDER")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(console_handler)

class KnowledgeExtractor:
    def __init__(self):
        try:
            local_llm.load_model()
        except Exception as e:
            logger.error(f"❌ Không thể load Local LLM: {e}")
            raise e

    def clean_json_response(self, text):
        """Làm sạch chuỗi JSON từ output của LLM (Robust Version)"""
        # 1. Loại bỏ thẻ <think> và markdown
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        
        # 2. Tìm khối JSON thô từ dấu { đầu tiên đến dấu } cuối cùng
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return "{}"
        
        json_str = text[start_idx : end_idx + 1]
        
        # 3. Fix lỗi phổ biến: Dấu phẩy thừa ở cuối list/dict (VD: {"a": 1,})
        # Regex này tìm dấu phẩy đứng trước dấu đóng ngoặc và xóa nó
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        return json_str.strip()

    def extract_graph_from_text(self, text_chunk):
        # One-shot Prompt: Cung cấp ví dụ cụ thể để định hướng model
        prompt = f"""
        You are a medical data extractor. Convert the text into a JSON Knowledge Graph.
        
        ### EXAMPLE:
        Text: "Metformin treats Type 2 Diabetes but may cause Nausea."
        JSON Output:
        {{
            "nodes": [
                {{"id": "Metformin", "label": "Drug"}},
                {{"id": "Type 2 Diabetes", "label": "Disease"}},
                {{"id": "Nausea", "label": "Symptom"}}
            ],
            "edges": [
                {{"source": "Metformin", "target": "Type 2 Diabetes", "type": "TREATS"}},
                {{"source": "Metformin", "target": "Nausea", "type": "CAUSES"}}
            ]
        }}
        
        ### TASK:
        Text: "{text_chunk}"
        
        Required: Output VALID JSON only. No explanations.
        """
        
        try:
            # --- TỐI ƯU GỌI HÀM ---
            messages = [
                {"role": "system", "content": "You are a JSON extractor. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            input_ids = local_llm.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(local_llm.model.device)
            
            attention_mask = import_torch().ones_like(input_ids)
            
            print("   ↳ 🤖 AI đang suy nghĩ...", end="\r")
            
            with import_torch().no_grad():
                outputs = local_llm.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=local_llm.tokenizer.pad_token_id,
                    max_new_tokens=1024, # Đủ dài cho JSON
                    temperature=0.1,     # Thấp để ổn định
                    do_sample=False      # Greedy search
                )
            
            raw_response = local_llm.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            print("   ↳ ✅ AI đã trả lời!       ") 

            clean_json = self.clean_json_response(raw_response)
            
            # --- PARSE JSON ---
            try:
                graph_data = json.loads(clean_json)
            except json.JSONDecodeError as e:
                print(f"   ↳ ⚠️ JSON Parse Error: {str(e)[:50]}")
                # print(f"DEBUG: {clean_json}") # Uncomment để debug
                return None
            
            # Chuẩn hóa keys
            if "nodes" not in graph_data: graph_data["nodes"] = []
            if "edges" not in graph_data: graph_data["edges"] = []
            
            return graph_data
            
        except Exception as e:
            print(f"   ↳ ❌ Lỗi hệ thống: {str(e)[:50]}...")
            return None

    def ingest_to_neo4j(self, graph_data):
        if not graph_data or not db_connector: return

        # Đảm bảo source/target trong edges đều tồn tại trong nodes để tránh lỗi orphan edges
        # Trong thực tế, có thể cần merge nodes trước
        
        node_query = """
        UNWIND $nodes AS n MERGE (node {name: n.id}) 
        ON CREATE SET node.id = n.id, node.source='User_Upload' 
        WITH node, n CALL apoc.create.addLabels(node, [n.label]) YIELD node as l RETURN count(l)
        """
        edge_query = """
        UNWIND $edges AS e MATCH (s {name: e.source}), (t {name: e.target}) 
        MERGE (s)-[r:RELATED {type: e.type, provenance:'User_Upload'}]->(t) RETURN count(r)
        """

        try:
            if graph_data.get("nodes"):
                db_connector.run_query(node_query, {"nodes": graph_data["nodes"]})
            if graph_data.get("edges"):
                db_connector.run_query(edge_query, {"edges": graph_data["edges"]})
            logger.info(f"   + DB: Saved {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges.")
        except Exception as e:
            logger.error(f"❌ DB Error: {e}")

def import_torch():
    import torch
    return torch

def main():
    print("🚀 Bắt đầu nạp dữ liệu (Robust Mode)")
    if db_connector is None: 
        print("❌ Không có kết nối DB.")
        return

    files = glob.glob(str(DATA_DIR / "*.txt"))
    if not files:
        print("⚠️ Không tìm thấy file .txt nào trong data/custom_knowledge")
        print("   -> Tạo file sample...")
        sample_file = DATA_DIR / "sample_vn.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("Cây chó đẻ (Diệp hạ châu) hỗ trợ trị viêm gan B nhưng gây hạ huyết áp.")
        files = [str(sample_file)]

    extractor = KnowledgeExtractor()
    
    for file_path in files:
        logger.info(f"📂 File: {os.path.basename(file_path)}")
        with open(file_path, "r", encoding="utf-8") as f: text = f.read()
        
        # Chia nhỏ text
        chunk_size = 800 
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for idx, chunk in enumerate(chunks):
            logger.info(f"   Processing chunk {idx+1}/{len(chunks)}...")
            graph_data = extractor.extract_graph_from_text(chunk)
            if graph_data: extractor.ingest_to_neo4j(graph_data)

    local_llm.unload()
    if db_connector: db_connector.close()
    print("\n🎉 Hoàn tất!")

if __name__ == "__main__":
    main()