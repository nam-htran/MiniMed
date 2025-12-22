# tests/test_step_5_reasoning.py
import numpy as np
from src.core.state import MedCOTState
from src.modules import step0_preprocess, step1_extraction, step2_linking, step4_retrieval, step5_reasoning
from src.utils.neo4j_connect import db_connector

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 5: GCoT REASONING")
    print("="*50)

    if db_connector is None:
        print("❌ Kết nối Neo4j thất bại. Dừng test.")
        return
        
    test_query = "What are the treatments for pterygium?"
    state = MedCOTState(raw_query=test_query)

    state = step0_preprocess.run(state)
    state = step1_extraction.run(state)
    state = step2_linking.run(state)
    state = step4_retrieval.run(state, top_k_nodes=50)

    if not state.graph_refs.get("ckg_subgraph", {}).get("nodes"):
        print("❌ Subgraph rỗng. Không thể chạy reasoning. Dừng test.")
        return
        
    # Chạy bước 5 với 2 bước suy luận
    state = step5_reasoning.run(state, num_think_steps=2)

    print("\n✅ KẾT QUẢ:")
    thought_vectors = state.gcot.get('thought_vectors', [])
    print(f"🔸 Số lượng thought vectors đã sinh: {len(thought_vectors)}")
    if thought_vectors:
        print(f"🔸 Shape của thought vector đầu tiên: {np.array(thought_vectors[0]).shape}")

    final_embeddings = state.graph_refs.get('final_node_embeddings', {})
    print(f"🔸 Số loại node có embedding cuối cùng: {len(final_embeddings)}")

    # Test này giờ sẽ PASS vì code Step 5 đã có vòng lặp
    assert len(thought_vectors) == 2, "Phải sinh đủ số thought vectors"
    assert len(final_embeddings) > 0, "Phải có final node embeddings"
    
    if db_connector:
        db_connector.close()

    print("\n🎉 TEST BƯỚC 5 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

