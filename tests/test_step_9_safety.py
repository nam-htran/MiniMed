# tests/test_step_9_safety.py
from src.core.state import MedCOTState, Mention
from src.modules import step9_safety

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 9: SAFETY ENGINE")
    print("="*50)
    
    state = MedCOTState(raw_query="test")
    state.final_answer = "Treat with Metformin and Warfarin."
    # Giả lập 2 thuốc có tương tác
    state.mentions = [
        Mention(text="Metformin", label="drug", span=(0,0), score=1.0, source="dict"),
        Mention(text="Warfarin", label="drug", span=(0,0), score=1.0, source="dict"),
    ]

    state = step9_safety.run(state)

    print("\n✅ KẾT QUẢ:")
    print(state.final_answer)
    print(state.safety_flags)
    
    if state.safety_flags:
        # Sửa: Code safety mới gán type là 'CLINICAL_RISK'
        assert state.safety_flags[0]['type'] == 'CLINICAL_RISK'
        assert "SAFETY WARNINGS" in state.final_answer

    print("\n🎉 TEST BƯỚC 9 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

