import pandas as pd
import os

# Cấu hình đường dẫn
INPUT_FILE = "data/org/kg.csv"
OUTPUT_DIR = "data/primekg/import"

# Tạo thư mục output
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"⏳ Đang đọc file gốc: {INPUT_FILE} ...")
try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file {INPUT_FILE}")
    print("👉 Hãy chạy: wget -O data/org/kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620")
    exit(1)

# --- 1. CLEANING & INSPECTION ---
print(f"📊 Số dòng dữ liệu: {len(df)}")
print(f"🔍 Các cột trong file CSV: {list(df.columns)}")

df.columns = df.columns.str.strip()
required_cols = ['x_id', 'x_type', 'x_name', 'y_id', 'y_type', 'y_name', 'relation']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"❌ Lỗi: File CSV thiếu các cột quan trọng: {missing_cols}")
    exit(1)

# --- 2. XỬ LÝ NODES (TẠO nodes.csv) ---
print("🔨 Đang xử lý Nodes...")
nodes_x = df[['x_id', 'x_type', 'x_name', 'x_source']].rename(columns={
    'x_id': ':ID', 'x_type': ':LABEL', 'x_name': 'name', 'x_source': 'source'
})
nodes_y = df[['y_id', 'y_type', 'y_name', 'y_source']].rename(columns={
    'y_id': ':ID', 'y_type': ':LABEL', 'y_name': 'name', 'y_source': 'source'
})
all_nodes = pd.concat([nodes_x, nodes_y], ignore_index=True)
all_nodes.drop_duplicates(subset=[':ID'], inplace=True)
all_nodes[':LABEL'] = all_nodes[':LABEL'].apply(lambda x: str(x).title())
nodes_path = os.path.join(OUTPUT_DIR, "nodes.csv")
all_nodes.to_csv(nodes_path, index=False)
print(f"✅ Đã lưu {len(all_nodes)} nodes vào: {nodes_path}")


# --- 3. XỬ LÝ EDGES (PHIÊN BẢN ĐẦY ĐỦ) ---
print("🔨 Đang xử lý Edges (Full Properties)...")

# Chuẩn bị cột pubmed_id: Neo4j-admin cần biết kiểu dữ liệu là mảng
# Ta thay thế dấu phẩy cách bằng dấu chấm phẩy để neo4j-admin tự tách mảng
if 'pubmed_id' in df.columns:
    df['pubmed_id'] = df['pubmed_id'].astype(str).str.replace(',', ';')

# Đổi tên cột, thêm các cột bằng chứng khoa học
edges = df.rename(columns={
    'x_id': ':START_ID',
    'y_id': ':END_ID',
    'relation': ':TYPE',
    'display_relation': 'display_relation',
    # === NÂNG CẤP ===
    'pubmed_id': 'pubmed_ids:string[]', # Chỉ định đây là mảng string cho neo4j-admin
    'evidence': 'evidence:string',
    'negation': 'negation:string'
})

# Chỉ lấy các cột cần thiết, bao gồm cả các cột mới
cols_to_keep = [
    ':START_ID', 
    ':END_ID', 
    ':TYPE', 
    'display_relation',
    'pubmed_ids:string[]', # Tên cột mới
    'evidence:string',     # Tên cột mới
    'negation:string'      # Tên cột mới
]

# Lọc bỏ các cột không tồn tại trong DataFrame để tránh lỗi
existing_cols_to_keep = [col for col in cols_to_keep if col in edges.columns]
edges = edges[existing_cols_to_keep]

# Chuẩn hóa Type quan hệ
edges[':TYPE'] = edges[':TYPE'].str.upper().str.replace(' ', '_')

# Lưu file edges.csv
edges_path = os.path.join(OUTPUT_DIR, "edges.csv")
edges.to_csv(edges_path, index=False)
print(f"✅ Đã lưu {len(edges)} edges (với đầy đủ thuộc tính) vào: {edges_path}")

print("🎉 PREPROCESSING HOÀN TẤT!")