# src/modules/step7_verification.py
import logging
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from pathlib import Path

from src.core.state import MedCOTState
from src.core import config
from src.models.verifier import MultiSignalVerifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("step7_verification_provenance")
_resources = {}
VERIFIER_MODEL_PATH = Path("models/verifier_weights.pth")

# --- NÂNG CẤP THEO CHỐT 7: THỨ TỰ ƯU TIÊN NGUỒN GỐC ---
PROVENANCE_SCORES = {
    "PSG": 1.0,           # Patient State Graph (Sự thật của ca bệnh)
    "PrimeKG": 0.85,      # KG Local, đã được curate
    "ARAX/KG2": 0.6,      # KG từ xa, nguồn đa dạng
    "User_Upload": 0.9,   # Dữ liệu do người dùng nạp, ưu tiên
    "DEFAULT": 0.3        # Mặc định nếu không có nguồn
}

def load_resources():
    global _resources
    if _resources: return _resources
    _resources['nli_model'] = CrossEncoder(config.NLI_MODEL_NAME)
    # --- NÂNG CẤP: INPUT_DIM TĂNG TỪ 7 LÊN 8 ĐỂ THÊM PROVENANCE ---
    _resources['verifier_model'] = MultiSignalVerifier(input_dim=8)
    if VERIFIER_MODEL_PATH.exists():
        _resources['verifier_model'].load_state_dict(torch.load(VERIFIER_MODEL_PATH))
    _resources['verifier_model'].eval()
    return _resources

def _get_node_meta(node_id, state):
    for node in state.graph_refs.get("ckg_subgraph", {}).get("nodes", []):
        if node["id"] == node_id: return node
    return None

def _extract_path_features(path, state, nli_model):
    path_features = []
    node_map = {n['id']: n for n in state.graph_refs.get("ckg_subgraph", {}).get("nodes", [])}
    
    for step in path:
        src_meta = node_map.get(step['source'])
        tgt_meta = node_map.get(step['target'])
        if not src_meta or not tgt_meta: continue

        step_text = f"{src_meta['name']} {step.get('edge_text', step['edge'])} {tgt_meta['name']}"
        try:
            scores = nli_model.predict([(state.normalized_query, step_text)])
            probs = torch.softmax(torch.tensor(scores), dim=-1)
            nli_score = float(probs[0][-1]) # Lấy điểm của "entailment"
        except Exception: nli_score = 0.5

        # --- NÂNG CẤP: THÊM TÍN HIỆU PROVENANCE VÀO VECTOR ---
        provenance_score = PROVENANCE_SCORES.get(step.get("provenance", "DEFAULT"), 0.3)
        
        # [nli, gcot, in_kg, causality, len, src_deg, tgt_deg, provenance]
        features = [ nli_score, 0.5, 1.0, 0.5, len(path), 1, 1, provenance_score ]
        path_features.append(features)
        
    return np.mean(path_features, axis=0) if path_features else None

def run(state: MedCOTState) -> MedCOTState:
    if not state.candidate_paths:
        state.reasoning_mode = "Abstain"
        return state

    resources = load_resources()
    path_vectors, valid_candidates = [], []
    for cand in state.candidate_paths:
        feats = _extract_path_features(cand['path'], state, resources['nli_model'])
        if feats is not None:
            path_vectors.append(feats)
            valid_candidates.append(cand)
            
    if not path_vectors:
        state.reasoning_mode = "Abstain"
        return state

    with torch.no_grad():
        logits = resources['verifier_model'](torch.tensor(np.array(path_vectors), dtype=torch.float32))
        confidences = torch.sigmoid(logits).squeeze().cpu().numpy()
        if np.ndim(confidences) == 0: confidences = [float(confidences)]

    for i, cand in enumerate(valid_candidates):
        cand['verification_confidence'] = float(confidences[i])

    verified_results = sorted(valid_candidates, key=lambda x: x['verification_confidence'], reverse=True)
    
    if verified_results and verified_results[0]['verification_confidence'] > 0.5:
        best = verified_results[0]
        state.verified_path = best['path']
        state.global_confidence = best['verification_confidence']
        state.reasoning_mode = "Graph-Strict" if state.global_confidence > 0.8 else "Cautious"
        state.gcot['verified_path_text'] = best.get('text_repr', '')
    else:
        state.reasoning_mode = "Abstain"
        
    state.log("7_VERIFICATION", "SUCCESS", {"mode": state.reasoning_mode, "conf": state.global_confidence})
    return state