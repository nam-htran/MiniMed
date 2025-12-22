# tests/test_step_2_linking.py
from src.core.state import MedCOTState
from src.modules import step0_preprocess, step1_extraction, step2_linking
from src.utils.neo4j_connect import db_connector

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 2: ENTITY LINKING")
    print("="*50)

    if db_connector is None:
        print("❌ Kết nối Neo4j thất bại. Dừng test.")
        return

    # Dùng test case tiếng Anh để đảm bảo có trong CKG dump
    test_query = "A patient with hypertension was treated with lisinopril."
    state = MedCOTState(raw_query=test_query)

    print(f"🔹 Query: '{test_query}'")

    # Chạy các bước phụ thuộc
    state = step0_preprocess.run(state, enable_phi_redaction=False)
    state = step1_extraction.run(state)
    
    print(f"🔸 Đã trích xuất {len(state.mentions)} mentions.")

    # Chạy bước 2
    state = step2_linking.run(state)

    print("\n✅ KẾT QUẢ:")
    for le in state.linked_entities:
        mention = le.source_mention
        if le.link_status == 'linked':
            best = le.best_candidate
            print(f"  [LINKED]   '{mention.text}' ({mention.kg_type}) -> {best.node_id} ('{best.preferred_name}')")
        else:
            print(f"  [UNLINKED] '{mention.text}' ({mention.kg_type})")

    print("\n🔸 Seed Nodes cuối cùng:")
    print(state.seed_nodes)

    assert len(state.seed_nodes) > 0, "Phải link được ít nhất 1 node"
    
    if db_connector:
        db_connector.close()
    print("\n🎉 TEST BƯỚC 2 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

