# Tệp: src/utils/umls_normalizer.py (PHIÊN BẢN CHUẨN ĐỂ SỬ DỤNG)
import logging
import sqlite3
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache

# --- CONFIG ---
SAB_RANKING = {
    "RXNORM": 1, "SNOMEDCT_US": 2, "NCI": 3, "MSH": 4,
    "HGNC": 5, "GO": 6, "MDR": 7
}
logger = logging.getLogger("UMLS_NORMALIZER")

class UMLSNormalizer:
    _instance = None
    
    def __new__(cls, db_path="data/umls/umls_lookup.db"):
        if cls._instance is None:
            cls._instance = super(UMLSNormalizer, cls).__new__(cls)
            cls._instance.db_path = Path(db_path)
            cls._instance.conn = None
        return cls._instance

    def connect(self):
        if self.conn is None and self.db_path.exists():
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row
                logger.info(f"✅ Đã kết nối tới cơ sở dữ liệu UMLS tại: {self.db_path}")
            except sqlite3.Error as e:
                logger.error(f"❌ Lỗi kết nối SQLite: {e}")
                self.conn = None

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    @lru_cache(maxsize=1024)
    def normalize(self, text: str, target_stys: tuple = None, top_k: int = 5) -> list[dict]:
        if not self.conn: return []
        text_lower = text.lower()
        query = "SELECT a.cui, a.str, a.is_pref, a.sab, a.tty, s.sty FROM atoms a LEFT JOIN semantic_types s ON a.cui = s.cui WHERE a.str_lower = ?"
        params = [text_lower]
        if target_stys:
            placeholders = ','.join('?' for _ in target_stys)
            query += f" AND s.sty IN ({placeholders})"
            params.extend(target_stys)
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except sqlite3.Error: return []
        if not rows: return []
        cui_candidates = {}
        for row in rows:
            cui = row['cui']
            if cui not in cui_candidates: cui_candidates[cui] = {"cui": cui, "atoms": [], "stys": set()}
            cui_candidates[cui]["atoms"].append(dict(row))
            if row['sty']: cui_candidates[cui]["stys"].add(row['sty'])
        scored_results = []
        for cui, data in cui_candidates.items():
            best_atom = sorted(data['atoms'], key=lambda x: (-x['is_pref'], SAB_RANKING.get(x['sab'], 99)))[0]
            score = (best_atom['is_pref'] * 100) + (10 - SAB_RANKING.get(best_atom['sab'], 99))
            scored_results.append({"cui": cui, "pref_name": best_atom['str'], "stys": list(data['stys']), "sab": best_atom['sab'], "score": score})
        return sorted(scored_results, key=lambda x: x['score'], reverse=True)[:top_k]

    # ==============================================================================
    # NÂNG CẤP: Thêm hàm lấy định nghĩa từ bảng definitions
    # ==============================================================================
    @lru_cache(maxsize=1024)
    def get_definition(self, cui: str) -> str:
        """
        Lấy định nghĩa của một CUI từ database.
        Trả về chuỗi định nghĩa hoặc chuỗi rỗng nếu không tìm thấy.
        """
        if not self.conn or not cui:
            return ""
        try:
            cursor = self.conn.cursor()
            # Lấy định nghĩa từ nguồn đáng tin cậy nhất (ưu tiên NCI)
            cursor.execute("""
                SELECT definition FROM definitions 
                WHERE cui = ? 
                ORDER BY CASE WHEN source = 'NCI' THEN 1 ELSE 2 END 
                LIMIT 1
            """, (cui,))
            row = cursor.fetchone()
            return row['definition'] if row else ""
        except sqlite3.Error as e:
            logger.error(f"Lỗi truy vấn `get_definition` cho CUI {cui}: {e}")
            return ""

# Khởi tạo singleton
umls_service = UMLSNormalizer()