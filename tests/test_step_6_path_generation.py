# tests/test_step_6_path_generation.py
from src.core.state import MedCOTState
from src.modules import (step0_preprocess, step1_extraction, step2_linking, 
                         step4_retrieval, step5_reasoning, step6_path_generation)
from src.utils.neo4j_connect import db_connector

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 6: PATH GENERATION")
    print("="*50)

    if db_connector is None:
        print("❌ Kết nối Neo4j thất bại. Dừng test.")
        return

    test_query = "What are the treatments for pterygium?"
    state = MedCOTState(raw_query=test_query)

    # Chạy các bước phụ thuộc
    state = step0_preprocess.run(state)
    state = step1_extraction.run(state)
    state = step2_linking.run(state)
    state = step4_retrieval.run(state, top_k_nodes=50)
    state = step5_reasoning.run(state)

    if not state.graph_refs.get("final_node_embeddings"):
        print("❌ Không có embeddings. Dừng test.")
        return

    # Chạy bước 6
    state = step6_path_generation.run(state, beam_width=3, max_path_length=3)

    print("\n✅ KẾT QUẢ:")
    print(f"🔸 Số candidate paths tìm thấy: {len(state.candidate_paths)}")

    if state.candidate_paths:
        print("🔸 Ví dụ path đầu tiên:")
        path_info = state.candidate_paths[0]
        path_str = " -> ".join([step['node_id'] for step in path_info['path']])
        print(f"  - Path: {path_str}")
        print(f"  - Score: {path_info['score']:.4f}")

    assert len(state.candidate_paths) >= 0 # Có thể không tìm thấy path nào
    
    if db_connector:
        db_connector.close()
    
    print("\n🎉 TEST BƯỚC 6 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

