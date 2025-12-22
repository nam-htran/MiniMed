# tests/test_step_1_extraction.py
from src.core.state import MedCOTState
from src.modules import step1_extraction
from pprint import pprint

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 1: HYBRID EXTRACTION")
    print("="*50)

    # Sử dụng query tiếng Anh để đảm bảo model hoạt động tốt nhất
    test_query = "The patient does not have fever, but has a history of hypertension and is taking metformin."
    state = MedCOTState(raw_query=test_query, normalized_query=test_query)

    print(f"🔹 Text đầu vào:\n'{state.normalized_query}'")

    # Chạy bước 1
    state = step1_extraction.run(state)

    print("\n✅ KẾT QUẢ:")
    print(f"🔸 Số thực thể tìm thấy: {len(state.mentions)}")
    for mention in state.mentions:
        print(f"  - Text: '{mention.text}', Label: {mention.label}, Attrs: {mention.attributes}")

    assert len(state.mentions) >= 2
    
    # Kiểm tra medspacy context (negation)
    fever_mention = next((m for m in state.mentions if "fever" in m.text.lower()), None)
    
    if fever_mention:
        # Code step 1 gán attrs['negated'] = True (thay vì negated_existence)
        is_neg = fever_mention.attributes.get('negated')
        print(f"  > 'fever' attributes: {fever_mention.attributes}")
        assert is_neg is True, "Fever should be negated"

    print("\n🎉 TEST BƯỚC 1 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

