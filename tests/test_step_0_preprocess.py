# tests/test_step_0_preprocess.py
from src.core.state import MedCOTState
from src.modules import step0_preprocess
from pprint import pprint

def main():
    print("="*50)
    print("🧪 BẮT ĐẦU TEST BƯỚC 0: PREPROCESSING")
    print("="*50)

    test_query = "   Bệnh nhân John Doe, 50 tuổi, có tiền sử ĐTĐ type 2.   \n\n Cần tư vấn thêm.  "
    state = MedCOTState(raw_query=test_query)

    print(f"🔹 Query gốc:\n'{state.raw_query}'")

    # Chạy bước 0
    state = step0_preprocess.run(state, enable_phi_redaction=True)

    print("\n✅ KẾT QUẢ:")
    print(f"🔸 Query đã chuẩn hóa (ẩn PHI):\n'{state.normalized_query}'")
    print("🔸 Các câu đã tách:")
    pprint(state.sentences)
    
    assert state.normalized_query == "Bệnh nhân <PERSON>, 50 tuổi, có tiền sử ĐTĐ type 2. \n\nCần tư vấn thêm."
    assert len(state.sentences) > 1

    print("\n🎉 TEST BƯỚC 0 THÀNH CÔNG!")

if __name__ == "__main__":
    main()

