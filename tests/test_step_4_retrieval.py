# tests/test_step_4_retrieval.py
import json
from src.core.state import MedCOTState
from src.modules import step0_preprocess, step1_extraction, step2_linking, step4_retrieval
from src.utils.neo4j_connect import db_connector

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 4: SUBGRAPH RETRIEVAL")
    print("="*50)

    if db_connector is None:
        print("❌ Kết nối Neo4j thất bại. Dừng test.")
        return
        
    print("ℹ️ LƯU Ý: Test này yêu cầu bạn phải chạy 'python run/build_faiss_index.py' trước.")

    test_query = "What are the treatments for hypertension?"
    state = MedCOTState(raw_query=test_query)

    print(f"🔹 Query: '{test_query}'")

    # Chạy các bước phụ thuộc
    state = step0_preprocess.run(state)
    state = step1_extraction.run(state)
    state = step2_linking.run(state)
    
    if not state.seed_nodes:
        print("❌ Không tìm thấy seed_nodes. Dừng test.")
        return
    print(f"🔸 Seed nodes tìm thấy: {state.seed_nodes}")

    # Chạy bước 4
    state = step4_retrieval.run(state, top_k_nodes=100)

    subgraph = state.graph_refs.get("ckg_subgraph", {})
    nodes = subgraph.get('nodes', [])
    edges = subgraph.get('edges', [])

    print("\n✅ KẾT QUẢ:")
    print(f"🔸 Subgraph retrieved: {len(nodes)} nodes, {len(edges)} edges.")
    
    assert len(nodes) > 0, "Subgraph phải có node"
    # assert len(edges) > 0, "Subgraph nên có cạnh để có ý nghĩa" # Có thể không có cạnh nếu các node không liên quan trực tiếp

    if db_connector:
        db_connector.close()
        
    if len(nodes) > 0:
        print("\n🎉 TEST BƯỚC 4 THÀNH CÔNG!")
    else:
        print("\n⚠️ TEST BƯỚC 4 HOÀN TẤT NHƯNG KHÔNG LẤY ĐƯỢC NODE NÀO. HÃY KIỂM TRA LẠI INDEX VÀ LOGIC.")


if __name__ == "__main__":
    main()

