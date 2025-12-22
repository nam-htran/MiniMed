# utils/neo4j_connect.py
import os
import time
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnection:
    """
    Quản lý kết nối Neo4j với cấu hình Timeout cao hơn và Retry.
    """
    def __init__(self, uri, user, password):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Driver = None
        self.connect()

    def connect(self):
        """Khởi tạo driver với cấu hình mạnh mẽ hơn."""
        # Nếu driver đã tồn tại, không tạo mới
        if self._driver is not None:
            return

        for i in range(3):
            try:
                self._driver = GraphDatabase.driver(
                    self._uri, 
                    auth=(self._user, self._password),
                    max_connection_lifetime=300,
                    keep_alive=True,
                    connection_acquisition_timeout=60,
                    connection_timeout=60
                )
                self._driver.verify_connectivity()
                print("✅ Kết nối Neo4j thành công!")
                return
            except Exception as e:
                print(f"⚠️ Lỗi kết nối lần {i+1}: {e}. Đang thử lại...")
                time.sleep(2)
        print("❌ Không thể kết nối Neo4j sau 3 lần thử.")

    def close(self):
        # --- FIX: Reset _driver về None sau khi đóng ---
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            print("🔌 Kết nối Neo4j đã đóng.")

    def run_query(self, query, parameters=None):
        if self._driver is None:
            self.connect()
            if self._driver is None: return []

        for attempt in range(3):
            try:
                with self._driver.session() as session:
                    result = session.run(query, parameters)
                    return list(result)
            except Exception as e:
                msg = str(e)
                if any(x in msg for x in ["ServiceUnavailable", "SessionExpired", "defunct", "Connection reset", "Closed"]):
                    print(f"⚠️ Connection drop detected ({msg}). Reconnecting ({attempt+1}/3)...")
                    self.close() # Reset driver
                    self.connect() # Re-init
                else:
                    print(f"❌ Query Error: {msg}")
                    raise e
        return []

# --- Singleton Instance ---
db_connector = None
try:
    db_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    db_user = os.getenv("NEO4J_USER", "neo4j")
    db_password = os.getenv("NEO4J_PASSWORD")

    if not db_password:
        print("⚠️ CẢNH BÁO: Biến môi trường NEO4J_PASSWORD chưa được thiết lập.")
    
    db_connector = Neo4jConnection(uri=db_uri, user=db_user, password=db_password)
except Exception as e:
    print(f">> LỖI NGHIÊM TRỌNG: Không thể khởi tạo kết nối database. {e}")
    db_connector = None