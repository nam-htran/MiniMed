# src/modules/step10_logging.py
import logging
import json
from pathlib import Path
from datetime import datetime, date
from src.core.state import MedCOTState
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("step10_logging")

def clean_for_json(obj):
    """
    Hàm đệ quy để làm sạch dữ liệu trước khi lưu vào JSON.
    Xử lý:
    1. Các kiểu số của Numpy (int64, float32, etc.) -> int, float của Python
    2. Numpy Arrays -> List
    3. Datetime/Date -> ISO String
    4. Các kiểu dữ liệu cơ bản -> Giữ nguyên
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    return obj

def run(state: MedCOTState, output_dir: str = "output/audit_logs") -> MedCOTState:
    logger.info("🚀 Bắt đầu ghi log và audit trail...")
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(output_dir) / f"{state.query_id}.json"
        
        # 1. Lấy dictionary thuần từ Pydantic
        state_dict = state.model_dump(mode='python')

        # 2. Làm sạch dữ liệu (Numpy + Datetime + các kiểu số đặc biệt)
        clean_state_dict = clean_for_json(state_dict)

        # 3. Ghi file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(clean_state_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✅ Đã lưu audit trail vào {log_file}")
        state.log("10_LOGGING", "SUCCESS", metadata={"log_file": str(log_file)})

    except Exception as e:
        logger.exception("Lỗi trong quá trình Logging")
        state.log("10_LOGGING", "FAILED", message=str(e))
        
    return state

