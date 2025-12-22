import os
import json
import requests
from pathlib import Path

# --- CẤU HÌNH ---
OUTPUT_DIR = Path("data/pubmedqa")
OUTPUT_FILE = OUTPUT_DIR / "test.jsonl"

# URL dữ liệu gốc từ GitHub của PubMedQA
URL_DATA = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
URL_TEST_SPLIT = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"

def download_file(url, save_path):
    """Hàm tải file từ URL"""
    if save_path.exists():
        print(f"⏩ File đã tồn tại: {save_path}")
        return
    
    print(f"⬇️ Đang tải {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Đã lưu: {save_path}")
    except Exception as e:
        print(f"❌ Lỗi tải file {url}: {e}")
        exit(1)

def format_context(contexts):
    """Nối các đoạn văn trong context thành một chuỗi duy nhất"""
    if isinstance(contexts, list):
        return " ".join(contexts)
    return str(contexts)

def main():
    # 1. Tạo thư mục data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_data_path = OUTPUT_DIR / "ori_pqal.json"
    test_split_path = OUTPUT_DIR / "test_ground_truth.json"

    # 2. Tải dữ liệu nguồn
    download_file(URL_DATA, raw_data_path)
    download_file(URL_TEST_SPLIT, test_split_path)

    # 3. Đọc dữ liệu
    print("🔄 Đang xử lý dữ liệu...")
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f) # Dictionary chứa toàn bộ 1k mẫu PQA-L
    
    with open(test_split_path, 'r', encoding='utf-8') as f:
        test_ids = json.load(f) # Dictionary {PMID: label} của tập test

    # 4. Chuyển đổi sang định dạng chuẩn JSONL
    # Format MedCOT cần: {"Question": ..., "Context": ..., "Correct Answer": ...}
    
    processed_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # Duyệt qua các ID nằm trong tập test chuẩn
        for pmid, label in test_ids.items():
            if pmid not in full_data:
                continue
                
            original_item = full_data[pmid]
            
            # Tạo record chuẩn hóa
            record = {
                "id": pmid,
                "Question": original_item["QUESTION"],
                # Context trong PubMedQA là list các câu, cần nối lại
                "Context": format_context(original_item["CONTEXTS"]), 
                "Correct Answer": label, # yes, no, hoặc maybe
                "Long Answer": original_item.get("LONG_ANSWER", ""),
                "Meshes": original_item.get("MESHES", [])
            }
            
            # Ghi dòng JSONL
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"🎉 Hoàn tất! Đã tạo file dataset tại: {OUTPUT_FILE}")
    print(f"📊 Tổng số mẫu Test: {processed_count}")

if __name__ == "__main__":
    main()