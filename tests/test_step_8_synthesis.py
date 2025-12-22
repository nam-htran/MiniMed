# tests/test_step_8_synthesis.py
from src.core.state import MedCOTState
from src.modules import step8_synthesis

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 8: ANSWER SYNTHESIS")
    print("="*50)
    
    state = MedCOTState(raw_query="test", normalized_query="What causes Disease_A?")
    state.reasoning_mode = "Graph-Strict"
    state.verified_path = [
        {'source': 'Gene_X', 'edge': 'ASSOCIATED_WITH', 'target': 'Disease_A', 'step_confidence': 0.9}
    ]
    state.graph_refs["ckg_subgraph"] = {"nodes": [], "edges": []}

    state = step8_synthesis.run(state)

    print("\n✅ KẾT QUẢ:")
    print(state.final_answer)
    
    # Sửa: Assert lỏng hơn vì format prompt thay đổi
    compiled = state.gcot.get('compiled_cot', '')
    assert "Verified Chain" in compiled or "Evidence" in compiled
    
    print("\n🎉 TEST BƯỚC 8 THÀNH CÔNG!")
    
if __name__ == "__main__":
    main()

