# tests/test_step_7_verification.py
from src.core.state import MedCOTState
from src.modules import (step0_preprocess, step1_extraction, step2_linking, 
                         step4_retrieval, step5_reasoning, step6_path_generation,
                         step7_verification)
from src.utils.neo4j_connect import db_connector

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 7: VERIFICATION")
    print("="*50)

    if db_connector is None: return

    test_query = "What are the treatments for hypertension?"
    state = MedCOTState(raw_query=test_query)

    # Chạy các bước phụ thuộc
    state = step0_preprocess.run(state)
    state = step1_extraction.run(state)
    state = step2_linking.run(state)
    state = step4_retrieval.run(state, top_k_nodes=50)
    state = step5_reasoning.run(state)
    state = step6_path_generation.run(state)

    if not state.candidate_paths:
        print("ℹ️ Không có candidate path. Dừng test.")
        return

    # Chạy bước 7
    state = step7_verification.run(state)

    print("\n✅ KẾT QUẢ:")
    print(f"🔸 Global Confidence: {state.global_confidence:.4f}")
    print(f"🔸 Reasoning Mode: {state.reasoning_mode}")
    
    # Sửa: Thêm 'Cautious' và 'Safety-Alert' vào danh sách hợp lệ
    valid_modes = ["Graph-Strict", "Cautious", "Cautious-Generic", "Abstain", "Safety-Alert"]
    assert state.reasoning_mode in valid_modes, f"Mode {state.reasoning_mode} không hợp lệ"
    
    if db_connector: db_connector.close()
    print("\n🎉 TEST BƯỚC 7 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

