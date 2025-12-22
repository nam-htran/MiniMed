import logging
import os
import re
from src.core.state import MedCOTState
# NÂNG CẤP: Import umls_service để có thể gọi hàm lấy định nghĩa
from src.utils.umls_normalizer import umls_service
from src.utils.local_llm import local_llm # Giả định dùng Local LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("step8_synthesis")

def clean_llm_output(text: str) -> str:
    """Loại bỏ các thẻ <think> và các thẻ XML khác khỏi output của LLM."""
    if not text: return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text, flags=re.DOTALL)
    return text.strip()

def run(state: MedCOTState) -> MedCOTState:
    # --- 1. Tổng hợp bằng chứng từ GRAPH (Giữ nguyên) ---
    evidence_lines = []
    node_map = {n['id']: n.get('name', n['id']) for n in state.graph_refs.get("ckg_subgraph", {}).get("nodes", [])}

    if state.verified_path:
        evidence_lines.append("✅ **Verified Reasoning Path:**")
        for step in state.verified_path:
            s_name = node_map.get(step['source'], step['source'])
            t_name = node_map.get(step['target'], step['target'])
            rel = step.get('edge_text', step.get('edge', 'related_to'))
            evidence_lines.append(f"- {s_name} --[{rel}]--> {t_name}")
    elif state.candidate_paths:
        evidence_lines.append("⚖️ **Potential Graph Connections:**")
        for p in state.candidate_paths[:5]:
            evidence_lines.append(f"- {p.get('text_repr', '')} (Score: {p.get('final_score', 0):.2f})")

    graph_evidence = "\n".join(evidence_lines)

    # ==============================================================================
    # NÂNG CẤP: Lấy định nghĩa từ UMLS để làm giàu ngữ cảnh cho LLM
    # ==============================================================================
    context_definitions = []
    processed_names = set()
    
    # Đảm bảo umls_service đã kết nối
    umls_service.connect()
    
    for le in state.linked_entities:
        if le.link_status == "linked" and le.best_candidate.preferred_name not in processed_names:
            name = le.best_candidate.preferred_name
            processed_names.add(name)
            
            # Dùng tên đã link để tìm lại CUI chuẩn nhất
            norm_results = umls_service.normalize(name, top_k=1)
            if norm_results:
                cui = norm_results[0].get('cui')
                definition = umls_service.get_definition(cui)
                if definition:
                    context_definitions.append(f"- **{name}:** {definition}")
    
    context_evidence = ""
    if context_definitions:
        context_evidence = "\n\n**Additional Context from Medical Dictionary:**\n" + "\n".join(context_definitions)
    # ==============================================================================

    # Ghép nối tất cả bằng chứng
    final_evidence_text = graph_evidence + context_evidence
    state.gcot['compiled_cot'] = final_evidence_text

    if not final_evidence_text.strip():
        state.final_answer = "I could not find sufficient evidence in the Knowledge Graph to answer this question."
        state.log("8_SYNTHESIS", "SKIPPED", "No evidence found")
        return state

    # --- 2. Xây dựng PROMPT đã được làm giàu ---
    prompt = f"""
    You are a medical AI assistant. Your task is to summarize the evidence into a direct answer.

    **USER QUESTION:** 
    {state.normalized_query}

    **PROVIDED EVIDENCE:**
    {final_evidence_text}

    **INSTRUCTIONS:**
    1.  Answer the user's question directly based **ONLY** on the "PROVIDED EVIDENCE".
    2.  If the evidence lists treatments, state them clearly.
    3.  If the evidence lists contraindications or warnings, state them first.
    4.  Do not add any information not present in the evidence.
    5.  Keep the answer concise and to the point.

    **Final Answer:**
    """

    # --- 3. THỰC THI (Sử dụng Local LLM) ---
    raw_answer = ""
    try:
        logger.info("⚡ Using Local LLM for synthesis with enriched context...")
        raw_answer = local_llm.generate_cot(prompt)
        state.final_answer = clean_llm_output(raw_answer)
        state.log("8_SYNTHESIS", "SUCCESS", {"model_used": "Local-LLM", "context_enriched": bool(context_definitions)})
    except Exception as e:
         logger.error(f"❌ Local LLM failed: {e}", exc_info=True)
         state.final_answer = f"**Raw Evidence found:**\n\n{final_evidence_text}"
         state.log("8_SYNTHESIS", "FAILED", {"error": str(e)})

    return state