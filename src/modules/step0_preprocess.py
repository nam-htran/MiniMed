# src/modules/step0_preprocess.py
import re
import unicodedata
import json
from pathlib import Path
from typing import Optional, Dict
import logging

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import spacy

from src.core.state import MedCOTState

# Giảm log thừa từ các thư viện
logging.getLogger("PyRuSH").setLevel(logging.WARNING)
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)

logger = logging.getLogger("step0_preprocess")

_resources = {}

def load_resources():
    """Tải các tài nguyên cần thiết cho preprocessing."""
    global _resources
    if _resources:
        return _resources
        
    logger.info("⏳ Loading Preprocessing resources...")
    try:
        # Tải từ điển viết tắt (nếu có)
        dict_path = Path("data/dictionaries/abbreviations.json")
        _resources["abbreviations"] = json.load(open(dict_path, "r", encoding="utf-8")) if dict_path.is_file() else {}
        
        # Khởi tạo Presidio để ẩn thông tin nhạy cảm (PHI)
        _resources["analyzer"] = AnalyzerEngine()
        _resources["anonymizer"] = AnonymizerEngine()
        _resources["phi_operators"] = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<PHI>"}),
        }
        
        # Tải model SpaCy để tách câu
        _resources["nlp"] = spacy.load("en_core_web_sm") # Dùng model tiếng Anh
        
    except Exception as e:
        logger.critical(f"❌ Failed to load preprocessing resources: {e}")
        # Thử tải lại model spacy nếu lỗi
        try:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _resources["nlp"] = spacy.load("en_core_web_sm")
        except Exception as spacy_e:
            logger.critical(f"❌ Spacy model download/load failed again: {spacy_e}")
            return {}
            
    return _resources

def _normalize_text(text: Optional[str]) -> str:
    """Chuẩn hóa text: bỏ khoảng trắng thừa, normalize unicode."""
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _expand_abbreviations(text: str, abbr_dict: Dict[str, str]) -> str:
    """Thay thế các từ viết tắt y khoa bằng dạng đầy đủ."""
    if not abbr_dict: return text
    for abbr, full_text in abbr_dict.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full_text, text, flags=re.IGNORECASE)
    return text

def run(state: MedCOTState, enable_phi_redaction: bool = True) -> MedCOTState:
    """
    Hàm chính của pipeline Step 0.
    Thực hiện: Chuẩn hóa, mở rộng viết tắt, ẩn thông tin nhạy cảm, tách câu.
    """
    resources = load_resources()
    if not resources.get("nlp") or not resources.get("analyzer"):
        state.log("0_PREPROCESS", "FAILED", "Resource loading failed.")
        raise RuntimeError("Preprocessing resources failed to load.")

    try:
        # 1. Xử lý Query
        normalized_query = _normalize_text(state.raw_query)
        expanded_query = _expand_abbreviations(normalized_query, resources.get("abbreviations", {}))
        
        redacted_query = expanded_query
        if enable_phi_redaction:
            analyzer_results = resources["analyzer"].analyze(text=expanded_query, language='en')
            redacted_query = resources["anonymizer"].anonymize(
                text=expanded_query, analyzer_results=analyzer_results, operators=resources["phi_operators"]
            ).text
        state.normalized_query = redacted_query

        # 2. Xử lý Patient Context
        if state.patient_context:
            normalized_context = _normalize_text(state.patient_context)
            expanded_context = _expand_abbreviations(normalized_context, resources.get("abbreviations", {}))
            
            redacted_context = expanded_context
            if enable_phi_redaction:
                analyzer_results = resources["analyzer"].analyze(text=expanded_context, language='en')
                redacted_context = resources["anonymizer"].anonymize(
                    text=expanded_context, analyzer_results=analyzer_results, operators=resources["phi_operators"]
                ).text
            state.normalized_patient_context = redacted_context
        
        # 3. Tách câu (chỉ tách câu của query chính)
        doc = resources["nlp"](state.normalized_query)
        state.sentences = [{"text": sent.text, "span": (sent.start_char, sent.end_char)} for sent in doc.sents]
        
        state.log("0_PREPROCESS", "SUCCESS", "Text normalized and sentences segmented.")
        
    except Exception as e:
        logger.exception("Error during preprocessing")
        state.log("0_PREPROCESS", "FAILED", message=str(e))
        raise e
        
    return state