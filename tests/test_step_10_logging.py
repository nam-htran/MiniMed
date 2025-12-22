# tests/test_step_10_logging.py
import os
from pathlib import Path
from src.core.state import MedCOTState
from src.modules import step10_logging

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 10: PROVENANCE LOGGING")
    print("="*50)
    
    state = MedCOTState(raw_query="final test")
    state.final_answer = "This is the final answer."
    state.global_confidence = 0.95

    print(f"🔹 Test với query_id: {state.query_id}")

    # Chạy bước 10
    state = step10_logging.run(state)
    
    log_file_path = Path("output/audit_logs") / f"{state.query_id}.json"

    print("\n✅ KẾT QUẢ:")
    print(f"🔸 File log dự kiến được tạo tại: {log_file_path}")
    
    assert log_file_path.exists(), f"File log {log_file_path} không được tạo!"
    
    # Dọn dẹp file test
    os.remove(log_file_path)
    print("🔸 File log test đã được xóa.")

    print("\n🎉 TEST BƯỚC 10 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

