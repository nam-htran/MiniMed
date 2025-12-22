import sqlite3
import logging
import os
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
# Script này giờ yêu cầu 5 file "vàng" từ UMLS Metathesaurus
MRCONSO_PATH = Path("data/umls/MRCONSO.RRF")
MRSTY_PATH = Path("data/umls/MRSTY.RRF")
MRDEF_PATH = Path("data/umls/MRDEF.RRF")
MRREL_PATH = Path("data/umls/MRREL.RRF")
MRSAT_PATH = Path("data/umls/MRSAT.RRF")
OUTPUT_DB_PATH = Path("data/umls/umls_lookup.db")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UMLS_BUILDER_ULTIMATE")

def build_db():
    if not OUTPUT_DB_PATH.parent.exists():
        OUTPUT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB_PATH.exists():
        logger.warning(f"Đã tìm thấy file DB cũ. Sẽ xóa và xây dựng lại.")
        os.remove(OUTPUT_DB_PATH)

    # Kiểm tra tất cả các file nguồn
    required_files = [MRCONSO_PATH, MRSTY_PATH, MRDEF_PATH, MRREL_PATH, MRSAT_PATH]
    if not all(f.exists() for f in required_files):
        logger.error("❌ Không tìm thấy đủ file nguồn UMLS! Cần có:")
        for f in required_files:
            logger.error(f"   - {f} {'(✅ TÌM THẤY)' if f.exists() else '(❌ KHÔNG TÌM THẤY)'}")
        return

    logger.info(f"🚀 Bắt đầu xây dựng cơ sở dữ liệu UMLS (Ultimate Version)...")
    conn = sqlite3.connect(str(OUTPUT_DB_PATH))
    cursor = conn.cursor()

    # Tối ưu hóa tốc độ ghi
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")

    logger.info("📦 Đang tạo cấu trúc bảng (Schema)...")
    cursor.execute('CREATE TABLE IF NOT EXISTS atoms (cui TEXT, str TEXT, str_lower TEXT, is_pref INTEGER, sab TEXT, tty TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS semantic_types (cui TEXT, tui TEXT, sty TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS definitions (cui TEXT, definition TEXT, source TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS relations (cui1 TEXT, rel_type TEXT, cui2 TEXT, source TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS attributes (cui TEXT, attr_name TEXT, attr_value TEXT, source TEXT)')
    conn.commit()

    # --- GIAI ĐOẠN 1/5: XỬ LÝ MRCONSO.RRF (Từ vựng) ---
    logger.info("⏳ GIAI ĐOẠN 1/5: Xử lý MRCONSO.RRF...")
    with open(MRCONSO_PATH, 'r', encoding='utf-8') as f:
        batch = []
        for line in tqdm(f, desc="Importing Concepts"):
            fields = line.strip().split('|')
            if len(fields) > 14 and fields[1] == 'ENG':
                batch.append((fields[0], fields[14], fields[14].lower(), 1 if fields[2] == 'P' else 0, fields[11], fields[12]))
            if len(batch) >= 100000:
                cursor.executemany("INSERT INTO atoms VALUES (?, ?, ?, ?, ?, ?)", batch); conn.commit(); batch = []
        if batch: cursor.executemany("INSERT INTO atoms VALUES (?, ?, ?, ?, ?, ?)", batch); conn.commit()

    # --- GIAI ĐOẠN 2/5: XỬ LÝ MRSTY.RRF (Loại thực thể) ---
    logger.info("⏳ GIAI ĐOẠN 2/5: Xử lý MRSTY.RRF...")
    with open(MRSTY_PATH, 'r', encoding='utf-8') as f:
        batch = []
        for line in tqdm(f, desc="Importing SemTypes"):
            fields = line.strip().split('|')
            if len(fields) > 3: batch.append((fields[0], fields[1], fields[3]))
            if len(batch) >= 100000:
                cursor.executemany("INSERT INTO semantic_types VALUES (?, ?, ?)", batch); conn.commit(); batch = []
        if batch: cursor.executemany("INSERT INTO semantic_types VALUES (?, ?, ?)", batch); conn.commit()

    # --- GIAI ĐOẠN 3/5: XỬ LÝ MRDEF.RRF (Định nghĩa) ---
    logger.info("⏳ GIAI ĐOẠN 3/5: Xử lý MRDEF.RRF...")
    with open(MRDEF_PATH, 'r', encoding='utf-8') as f:
        batch = []
        for line in tqdm(f, desc="Importing Definitions"):
            fields = line.strip().split('|')
            if len(fields) > 5: batch.append((fields[0], fields[5], fields[4]))
            if len(batch) >= 100000:
                cursor.executemany("INSERT INTO definitions VALUES (?, ?, ?)", batch); conn.commit(); batch = []
        if batch: cursor.executemany("INSERT INTO definitions VALUES (?, ?, ?)", batch); conn.commit()

    # --- GIAI ĐOẠN 4/5: XỬ LÝ MRREL.RRF (Quan hệ) ---
    logger.info("⏳ GIAI ĐOẠN 4/5: Xử lý MRREL.RRF...")
    with open(MRREL_PATH, 'r', encoding='utf-8') as f:
        batch = []
        for line in tqdm(f, desc="Importing Relations"):
            fields = line.strip().split('|')
            if len(fields) > 10: batch.append((fields[0], fields[7], fields[4], fields[10])) # CUI1, RELA, CUI2, SAB
            if len(batch) >= 100000:
                cursor.executemany("INSERT INTO relations VALUES (?, ?, ?, ?)", batch); conn.commit(); batch = []
        if batch: cursor.executemany("INSERT INTO relations VALUES (?, ?, ?, ?)", batch); conn.commit()
        
    # --- GIAI ĐOẠN 5/5: XỬ LÝ MRSAT.RRF (Thuộc tính) ---
    logger.info("⏳ GIAI ĐOẠN 5/5: Xử lý MRSAT.RRF...")
    with open(MRSAT_PATH, 'r', encoding='utf-8') as f:
        batch = []
        for line in tqdm(f, desc="Importing Attributes"):
            fields = line.strip().split('|')
            if len(fields) > 10: batch.append((fields[0], fields[8], fields[10], fields[4])) # CUI, ATN, ATV, SAB
            if len(batch) >= 100000:
                cursor.executemany("INSERT INTO attributes VALUES (?, ?, ?, ?)", batch); conn.commit(); batch = []
        if batch: cursor.executemany("INSERT INTO attributes VALUES (?, ?, ?, ?)", batch); conn.commit()

    # --- TẠO INDEX ---
    logger.info("🔨 Đang tạo Index để tra cứu nhanh...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_atoms_str_lower ON atoms (str_lower);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_atoms_cui ON atoms (cui);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sem_types_cui ON semantic_types (cui);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_defs_cui ON definitions (cui);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rels_cui1 ON relations (cui1);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rels_cui2 ON relations (cui2);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attrs_cui ON attributes (cui);")
    conn.commit()
    conn.close()

    db_size = OUTPUT_DB_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"✅✅✅ HOÀN TẤT! Đã tạo DB UMLS đầy đủ tại: {OUTPUT_DB_PATH}")
    logger.info(f"📊 Kích thước Database: {db_size:.2f} MB")

if __name__ == "__main__":
    build_db()