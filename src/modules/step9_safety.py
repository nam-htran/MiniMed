# Tệp: src/modules/step9_safety.py (Phiên bản cuối cùng, dựa trên ID từ seed_nodes)
import logging
from src.core.state import MedCOTState
from src.utils.neo4j_connect import db_connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("step9_safety")

def run(state: MedCOTState) -> MedCOTState:
    # --- SỬA ĐỔI DỨT ĐIỂM: SỬ DỤNG state.seed_nodes LÀ NGUỒN ID DUY NHẤT ---
    # state.seed_nodes đã được step4 cập nhật và chứa TẤT CẢ các ID liên quan (cả nội bộ và bên ngoài).
    query_entity_ids = set(state.seed_nodes)
    # --------------------------------------------------------------------
    
    # Xóa các safety flags cũ để chạy lại logic, tránh ghi đè sai
    state.safety_flags = []
    
    subgraph = state.graph_refs.get("ckg_subgraph", {})
    edges = subgraph.get("edges", [])
    nodes = {n['id']: n.get('name', 'Unknown') for n in subgraph.get("nodes", [])}

    logger.info(f"🛡️ Safety Check on {len(edges)} retrieved edges. Focusing on interactions between IDs: {query_entity_ids}")

    risk_keywords = ["INTERACT", "CONTRAINDICAT", "ADVERSE", "RISK", "SIDE_EFFECT", "AFFECTS"]
    
    direct_alerts = set()
    for edge in edges:
        rel_type = edge.get("type", "").upper()
        if any(risk in rel_type for risk in risk_keywords):
            source_id = edge.get('source')
            target_id = edge.get('target')
            
            # So sánh trực tiếp bằng ID
            if source_id in query_entity_ids and target_id in query_entity_ids:
                src_name = nodes.get(source_id, str(source_id))
                tgt_name = nodes.get(target_id, str(target_id))
                
                # Bỏ qua các tương tác không có ý nghĩa (ví dụ: Aspirin RELATED_TO Aspirin)
                if src_name.lower() == tgt_name.lower():
                    continue

                sorted_pair = tuple(sorted((src_name, tgt_name)))
                alert_msg = f"Direct Interaction Detected: {sorted_pair[0]} --[{rel_type}]--> {sorted_pair[1]}"
                direct_alerts.add(alert_msg)

    all_alerts = list(direct_alerts)

    # Fallback: Nếu không có tương tác trực tiếp, tìm cảnh báo chung
    if not all_alerts:
        logger.info("No direct interactions found. Looking for general contraindications for query entities.")
        other_alerts = []
        for edge in edges:
            rel_type = edge.get("type", "").upper()
            if "CONTRAINDICATION" in rel_type:
                 source_id = edge.get('source')
                 target_id = edge.get('target')
                 if source_id in query_entity_ids or target_id in query_entity_ids:
                    src_name = nodes.get(source_id, str(source_id))
                    tgt_name = nodes.get(target_id, str(target_id))
                    other_alerts.append(f"General Warning: {src_name} --[{rel_type}]--> {tgt_name}")
        all_alerts.extend(other_alerts[:5])

    # Gán cờ an toàn và chèn vào câu trả lời cuối cùng
    if all_alerts:
        state.reasoning_mode = "Safety-Alert"
        state.safety_flags = [{"type": "CLINICAL_RISK", "msg": msg} for msg in all_alerts]
        
        warning_block = "**🚨 SAFETY WARNING:**\n" + "\n".join(f"- {msg}" for msg in all_alerts)
        if state.final_answer:
            # Chỉ chèn vào nếu nó chưa tồn tại để tránh lặp lại
            if warning_block not in state.final_answer:
                state.final_answer = warning_block + "\n\n" + state.final_answer
        else:
            state.final_answer = warning_block

    state.log("9_SAFETY", "SUCCESS", {"alerts": len(all_alerts)})
    return state