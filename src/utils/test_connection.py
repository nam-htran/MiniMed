# test_connection.py
import pandas as pd
# Mới: Import thẳng đối tượng connector đã được khởi tạo sẵn
from src.utils.neo4j_connect import db_connector

def main_test():
    """
    Hàm test chính, sử dụng connector đã được tái cấu trúc.
    """
    # Mới: Kiểm tra xem connector có được tạo thành công không
    if db_connector is None:
        print("❌ Không thể chạy test vì kết nối database thất bại.")
        return

    print("\n--- Bắt đầu chạy test query ---")
    try:
        query = """
        MATCH (d:Disease)
        RETURN d.id AS ID, d.name AS Name
        LIMIT 5
        """
        # Mới: Chạy query cực kỳ đơn giản
        data = db_connector.run_query(query)
        
        if data:
            print(f"📊 Tìm thấy dữ liệu mẫu ({len(data)} records):")
            df = pd.DataFrame(data)
            print(df)
        else:
            print("⚠️ Không tìm thấy node :Disease nào.")
            
    except Exception as e:
        print(f"❌ Lỗi khi đang chạy query: {e}")
    finally:
        # Mới: Đóng kết nối (quan trọng khi ứng dụng kết thúc)
        if db_connector:
            db_connector.close()

if __name__ == "__main__":
    main_test()

