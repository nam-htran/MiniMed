# 🧠 MedCOT: Neuro-Symbolic Medical AI

**MedCOT** là một hệ thống Trí tuệ Nhân tạo Y khoa tiên tiến, được xây dựng theo kiến trúc Thần kinh-Biểu tượng (Neuro-Symbolic). Dự án kết hợp sức mạnh của **Đồ thị Tri thức (Knowledge Graph)** để lưu trữ các mối quan hệ y khoa một cách có cấu trúc và **Mô hình Ngôn ngữ Lớn (LLM)** để xử lý ngôn ngữ tự nhiên và suy luận.

Mục tiêu của MedCOT là cung cấp các câu trả lời cho những câu hỏi y khoa phức tạp, đảm bảo tính chính xác, có thể truy vết nguồn gốc (evidence-based), và tích hợp các cơ chế kiểm tra an toàn tự động.

## ✨ Tính năng nổi bật

-   **Trích xuất Thực thể Y khoa Lai:** Kết hợp model `GLiNER` và từ điển chuyên gia để nhận diện các thực thể như Bệnh, Thuốc, Triệu chứng.
-   **Đồ thị Tri thức Đa nguồn:** Tích hợp dữ liệu từ **PrimeKG** (với đầy đủ bằng chứng khoa học), **UMLS** (từ điển y khoa khổng lồ), và cho phép người dùng tự nạp thêm tri thức.
-   **Suy luận trên Đồ thị (Graph Reasoning):** Sử dụng Graph Neural Network (GNN) để tạo ra các vector nhúng (embeddings) giàu ngữ cảnh và thuật toán tìm đường đi để khám phá các mối liên hệ.
-   **Kiểm tra và Xác thực Bằng chứng:** Mỗi đường đi suy luận được đánh giá độ tin cậy thông qua một mô hình xác thực (Verifier Model) đa tín hiệu, bao gồm cả nguồn gốc dữ liệu (provenance).
-   **Engine Kiểm tra An toàn Tự động:** Tự động phát hiện các tương tác thuốc nguy hiểm (DDI) và chống chỉ định (Contraindications) dựa trên bằng chứng trong đồ thị.
-   **Tổng hợp Câu trả lời bằng LLM có Ngữ cảnh:** Sử dụng Local LLM (DeepSeek 1.5B) để tạo ra câu trả lời cuối cùng, với prompt được làm giàu bằng cả **bằng chứng từ đồ thị** và **định nghĩa khoa học từ UMLS**.
-   **Quy trình Training & Đánh giá Toàn diện:** Cung cấp đầy đủ script để tạo dataset, huấn luyện các model GNN và 3 kiến trúc LLM khác nhau (LoRA-Default, LoRA-MedCOT, TRM), và đánh giá hiệu năng so với các baseline như GPT-4o.
-   **Giao diện Demo trực quan:** Tích hợp giao diện web bằng Streamlit để dễ dàng sử dụng và trình bày kết quả.

## ⚙️ Luồng xử lý chi tiết của Pipeline

Khi nhận một câu hỏi, MedCOT sẽ thực thi một chuỗi 10 bước xử lý tuần tự:

> #### **Giai đoạn 1: Chuẩn bị & Ánh xạ Dữ liệu**
>
> 1.  **Step 0: Preprocessing:** Làm sạch và chuẩn hóa văn bản đầu vào (câu hỏi và ngữ cảnh bệnh nhân), tách câu, và ẩn các thông tin định danh cá nhân (PHI) nếu có.
> 2.  **Step 1: Entity Extraction:** Sử dụng model GLiNER và thư viện MedSpacy để trích xuất các thực thể y khoa quan trọng như `Bệnh`, `Thuốc`, `Triệu chứng` từ văn bản đã được làm sạch.
> 3.  **Step 2: Entity Linking:** Ánh xạ các thực thể vừa trích xuất được vào các `node` cụ thể trong Đồ thị Tri thức Neo4j. Quá trình này sử dụng tìm kiếm chính xác và mở rộng từ đồng nghĩa thông qua database UMLS. Các `node` được link thành công sẽ trở thành "hạt giống" (seed nodes) cho bước tiếp theo.

> #### **Giai đoạn 2: Suy luận & Tìm kiếm Bằng chứng**
>
> 4.  **Step 4: Subgraph Retrieval:** Từ các "hạt giống", truy vấn vào Neo4j để lấy ra một đồ thị con (subgraph) chứa các node liên quan và các cạnh nối giữa chúng. Đồng thời, hệ thống cũng gọi đến các API bên ngoài (ARAX) để làm giàu thêm các mối quan hệ chưa có trong đồ thị cục bộ.
> 5.  **Step 5: GNN Reasoning:** Đồ thị con và câu hỏi đầu vào được đưa vào một Graph Neural Network (GNN). GNN sẽ tính toán các vector nhúng (node embeddings) mới cho mỗi node, giúp các vector này "hiểu" được ngữ cảnh của câu hỏi hiện tại.
> 6.  **Step 6: Path Generation:** Dựa trên các vector nhúng đã được làm giàu, hệ thống thực hiện thuật toán tìm kiếm (Beam Search) để tìm ra các "đường đi suy luận" (ví dụ: `Thuốc A -> TREATS -> Bệnh B`) tiềm năng nhất để trả lời câu hỏi.
> 7.  **Step 7: Path Verification:** Mỗi đường đi suy luận được đưa vào một mô hình phân loại (Verifier Model) để đánh giá độ tin cậy. Mô hình này xem xét nhiều yếu tố như sự liên quan về ngữ nghĩa, nguồn gốc dữ liệu, cấu trúc đường đi để chọn ra đường đi được xác thực (verified path) tốt nhất.

> #### **Giai đoạn 3: Tổng hợp Câu trả lời & Kiểm tra An toàn**
>
> 8.  **Step 9: Safety Check (Lần 1):** Hệ thống quét nhanh đồ thị con để tìm kiếm các mối quan hệ nguy hiểm như `INTERACTS_WITH` (tương tác thuốc) hoặc `CONTRAINDICATION` (chống chỉ định) liên quan đến các thực thể trong câu hỏi.
> 9.  **Step 8: Answer Synthesis:** Một prompt chi tiết được xây dựng, bao gồm: câu hỏi gốc, đường đi suy luận đã được xác thực, và **các định nghĩa khoa học** của thực thể (lấy từ UMLS). Prompt này được đưa cho Local LLM để tạo ra một câu trả lời hoàn chỉnh, có giải thích.
> 10. **Step 9: Safety Check (Lần 2):** Nếu các cờ an toàn được phát hiện ở lần quét đầu, một khối cảnh báo đặc biệt sẽ được định dạng và chèn vào **đầu** câu trả lời cuối cùng để người dùng không thể bỏ lỡ.

> #### **Giai đoạn 4: Lưu trữ & Truy vết**
>
> 11. **Step 10: Logging:** Toàn bộ quá trình xử lý, từ đầu vào, các kết quả trung gian, đến câu trả lời cuối cùng, được lưu vào một file JSON duy nhất. File log này phục vụ cho việc gỡ lỗi, kiểm tra và đảm bảo tính minh bạch của hệ thống.

## 🚀 Hướng dẫn Cài đặt & Khởi chạy (Dành cho Người dùng)

Thực hiện chính xác các bước sau để dựng và chạy hệ thống ở chế độ **sử dụng (inference)**.

### 1. Yêu cầu Tiên quyết

-   **Git:** Để clone source code.
-   **Docker & Docker Compose:** Để chạy database Neo4j. Đảm bảo Docker Desktop đang chạy.
-   **Conda/Miniconda:** Để quản lý môi trường Python.
-   **Đối với Windows:** **Git Bash** là bắt buộc để chạy các file script `.sh`.

### 2. Cài đặt Môi trường & Thư viện

```bash
# 1. Clone a new repository
git clone [<your-repository-url>](https://github.com/nam-htran/MiniMed)
cd MiniMed

# 2. Tạo môi trường Conda
conda create -n medcot python=3.11 -y
conda activate medcot

# 3. Cài đặt các thư viện Python cần thiết
pip install -r requirements.txt

# 4. (RẤT QUAN TRỌNG) Tải model ngôn ngữ cho SpaCy
# Pipeline cần model này để tách câu và xử lý văn bản ở bước đầu tiên.
python -m spacy download en_core_web_sm
```
**Lưu ý:** Nếu lệnh `spacy download` báo lỗi 404, hãy nâng cấp phiên bản `spacy` của bạn bằng lệnh `pip install --upgrade spacy` rồi thử lại.

### 3. Tải và Sắp xếp Dữ liệu Nguồn

Bạn cần tải thủ công các bộ dữ liệu lớn và đặt chúng vào đúng cấu trúc thư mục sau:

```
data/
├── org/
│   └── kg.csv             # <-- 1. Tải PrimeKG từ Harvard Dataverse
└── umls/
    ├── MRCONSO.RRF        # \
    ├── MRSTY.RRF          #  \
    ├── MRDEF.RRF          #   >-- 2. Lấy 5 file này từ UMLS Metathesaurus
    ├── MRREL.RRF          #  /
    └── MRSAT.RRF          # /
```
1.  **PrimeKG:** Tải file `kg.csv` từ [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM).
2.  **UMLS:** Yêu cầu tài khoản UMLS. Tải bộ Metathesaurus Full Release, giải nén và tìm 5 file `.RRF` ở trên.

### 4. Xây dựng Database, Index và Khởi động Hệ thống

Chạy lần lượt các script sau từ thư mục gốc dự án. **Thứ tự rất quan trọng!**

```bash
# Bước 4.1: Xây dựng Database UMLS từ file RRF (sẽ mất rất nhiều thời gian)
echo "--- Bắt đầu xây dựng UMLS Database ---"
python scripts/build_umls_db.py

# Bước 4.2: Chuẩn hóa dữ liệu PrimeKG cho Neo4j (phiên bản đầy đủ)
echo "--- Bắt đầu chuẩn hóa PrimeKG ---"
python scripts/0_preprocess_primekg.py

# Bước 4.3: Nạp dữ liệu vào Neo4j và khởi động server
# (Trên Windows, hãy đảm bảo bạn đang chạy trong GIT BASH)
echo "--- Bắt đầu nạp và khởi động Neo4j ---"
bash scripts/setup_import_primekg.sh

# Bước 4.4: Xây dựng Vector Index (FAISS) để tìm kiếm
# (Chỉ chạy sau khi Neo4j đã khởi động thành công ở bước trên)
echo "--- Bắt đầu xây dựng FAISS Index ---"
python scripts/2_build_faiss.py

echo "--- HOÀN TẤT CÀI ĐẶT! ---"
```

### 5. Chạy Pipeline

Sau khi tất cả các bước trên hoàn tất, bạn có thể bắt đầu sử dụng hệ thống.

**Cách 1: Chạy bằng Dòng lệnh (CLI)**

```bash
# Câu hỏi đơn giản
python main.py --query "What are the treatments for hypertension?"

# Câu hỏi phức tạp hơn với ngữ cảnh bệnh nhân
python main.py --query "Can the patient take metformin?" --context "The patient has a history of severe kidney disease."
```

**Cách 2: Chạy Giao diện Web (Streamlit)**

```bash
streamlit run app_demo.py
```
Sau đó, mở trình duyệt và truy cập `http://localhost:8501`.

---

## 🎓 Hướng dẫn Nâng cao (Dành cho Nhà phát triển: Training)

Phần này hướng dẫn cách huấn luyện lại các mô hình GNN và LLM của dự án từ đầu.

### Giai đoạn 1: Tạo Dataset Huấn luyện

Bước đầu tiên là chạy pipeline MedCOT trên một bộ dữ liệu câu hỏi-trả lời có sẵn (`.parquet`) để sinh ra các "dấu vết suy luận" (reasoning traces).

```bash
# Chạy script để tạo file data/medcot_rich_training_data.jsonl
python scripts/1_generate_dataset.py
```
> **Lưu ý:** Script này yêu cầu file `data/medical_o1_vi_translated_EVALUATED_GEMINI.parquet`. Bạn cần thay thế bằng file dataset của riêng mình và cập nhật đường dẫn trong script.

### Giai đoạn 2: Huấn luyện các Model phụ trợ

**2.1 Huấn luyện GNN**

```bash
# 1. Chuẩn bị dữ liệu GNN từ file .jsonl đã tạo
python scripts/3_prepare_gnn.py

# 2. Huấn luyện model GNN
python scripts/4_train_gnn.py
```
*Kết quả:* File trọng số `models/gnn_dual_tower_weights.pth` sẽ được tạo/cập nhật.

**2.2 Huấn luyện Verifier**
```bash
# Script này sử dụng dữ liệu giả lập để huấn luyện
python scripts/train_aux_verifier.py
```
*Kết quả:* File trọng số `models/verifier_weights.pth` sẽ được tạo/cập nhật.

### Giai đoạn 3: Huấn luyện Mô hình Ngôn ngữ Lớn (LLM)

Dự án cung cấp 3 phương pháp huấn luyện LLM khác nhau, được quản lý qua các file cấu hình `YAML` trong thư mục `configs/`.

1.  **LoRA on Default CoT (Baseline):**
    *   **Mục đích:** Huấn luyện một model baseline, chỉ sử dụng Chain-of-Thought (CoT) mặc định có sẵn trong dataset.
    *   **Cấu hình:** `configs/sft_default_config.yaml` (bạn cần tạo file này nếu chưa có)
    *   **Lệnh:** `python scripts/5_train_llm.py --config configs/sft_default_config.yaml`

2.  **LoRA on MedCOT CoT (Phương pháp chính):**
    *   **Mục đích:** Huấn luyện model sử dụng các dấu vết suy luận chất lượng cao được sinh ra từ pipeline đồ thị (MedCOT). Đây là phương pháp cốt lõi của dự án.
    *   **Cấu hình:** `configs/sft_medcot_config.yaml` (bạn cần tạo file này nếu chưa có)
    *   **Lệnh:** `python scripts/5_train_llm.py --config configs/sft_medcot_config.yaml`

3.  **TRM-inspired Model (Phương pháp nâng cao):**
    *   **Mục đích:** Huấn luyện model với một prompt phức tạp hơn, dạy nó cách "tự nâng cao" (self-enhance) dấu vết suy luận trước khi đưa ra câu trả lời.
    *   **Cấu hình:** `configs/sft_trm_config.yaml` (bạn cần tạo file này nếu chưa có)
    *   **Lệnh:** `python scripts/5_train_llm.py --config configs/sft_trm_config.yaml`

*Kết quả:* Các adapter LoRA sẽ được lưu vào thư mục được chỉ định trong file config (ví dụ: `models/sft_medcot_adapter`).

### Giai đoạn 4: Đánh giá Model

Sau khi huấn luyện, bạn có thể chạy script đánh giá để so sánh hiệu năng của các model mới với các baseline (GPT-4o, RAG, etc.) trên bộ dữ liệu test (ví dụ: PubMedQA).

```bash
# 1. Chuẩn bị dataset PubMedQA (nếu cần)
python scripts/prepare_pubmedqa.py

# 2. Chỉnh sửa configs/evaluate_config.yaml để thêm model mới của bạn

# 3. Chạy script đánh giá
python scripts/6_evaluate_models.py --config configs/evaluate_config.yaml
```
*Kết quả:* Một file `evaluation_results_pubmedqa.csv` chứa điểm số và output của từng model sẽ được tạo ra.

## 📁 Cấu trúc Thư mục

```
.
├── configs/           # Các file YAML cấu hình cho việc training và evaluation
├── data/              # Chứa dữ liệu thô và đã xử lý (UMLS, PrimeKG, FAISS index)
├── scripts/           # Các script để xây dựng database, index, và training model
├── src/               # Toàn bộ source code của pipeline MedCOT
│   ├── core/          # Định nghĩa State, Config cốt lõi
│   ├── modules/       # Mỗi file là một bước trong pipeline (Step 0 -> Step 10)
│   └── utils/         # Các tiện ích kết nối (Neo4j, UMLS, LLM...)
├── tests/             # Các file unit test cho từng module
├── app_demo.py        # Giao diện web Streamlit
├── main.py            # Điểm khởi chạy chính của pipeline (CLI)
├── docker-compose.yml # Cấu hình để chạy Neo4j
└── README.md          # File này
```
