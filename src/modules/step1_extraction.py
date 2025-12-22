import logging
from typing import List, Dict, Any

# Di chuyển các import nặng vào trong hàm để kiểm soát lỗi tốt hơn
# from gliner import GLiNER
# import medspacy

from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.matcher import PhraseMatcher

from src.core.state import MedCOTState, Mention
from src.core import config

logger = logging.getLogger("step1_hybrid_extraction")

_models = {}
_models_loaded = False

def load_models_bulletproof():
    global _models, _models_loaded
    if _models_loaded:
        return _models

    logger.info("--- Starting Resource Loading for Step 1 ---")
    
    # 1. Tải GLiNER
    try:
        from gliner import GLiNER
        logger.info("  -> Attempting to load GLiNER model...")
        _models["gliner"] = GLiNER.from_pretrained(config.EXTRACTION_MODEL_NAME)
        logger.info("  [SUCCESS] GLiNER model loaded.")
    except Exception as e:
        logger.error(f"  [CRITICAL FAILURE] Could not load GLiNER model. Extraction will be degraded. Error: {e}", exc_info=True)
        _models["gliner"] = None

    # 2. Tải MedSpaCy
    try:
        import medspacy
        logger.info("  -> Attempting to load MedSpaCy model...")
        nlp = medspacy.load(enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])
        if not Span.has_extension("source_mention"):
            Span.set_extension("source_mention", default=None)
        _models["medspacy"] = nlp
        logger.info("  [SUCCESS] MedSpaCy model loaded.")
    except Exception as e:
        logger.error(f"  [CRITICAL FAILURE] Could not load MedSpaCy model. Context analysis disabled. Error: {e}", exc_info=True)
        _models["medspacy"] = None

    # 3. Xây dựng PhraseMatcher (nhẹ, ít khi lỗi)
    if _models.get("medspacy"):
        try:
            logger.info("  -> Building PhraseMatcher...")
            nlp = _models["medspacy"]
            matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            for term, (label, kg_type) in config.KNOWN_ENTITIES.items():
                pattern_id = f"{label}||{kg_type}"
                matcher.add(pattern_id, [nlp.make_doc(term)])
            _models["matcher"] = matcher
            logger.info("  [SUCCESS] PhraseMatcher built.")
        except Exception as e:
            logger.error(f"  [FAILED] Could not build PhraseMatcher. Error: {e}")
            _models["matcher"] = None
    
    _models_loaded = True
    logger.info("--- Resource Loading for Step 1 Finished ---")
    return _models

def _run_ner_on_text(text: str, models: dict) -> List[Dict[str, Any]]:
    if not text: return []
    
    gliner_ents = []
    if models.get("gliner"):
        try:
            raw_preds = models["gliner"].predict_entities(text, config.ENTITY_LABELS, threshold=config.DEFAULT_EXTRACTION_THRESHOLD)
            for e in raw_preds:
                gliner_ents.append({
                    "text": e["text"], 
                    "label": config.GLINER_TO_INTERNAL_LABEL_MAP.get(e["label"], "unknown"), 
                    "span": (e["start"], e["end"]), 
                    "score": e["score"], 
                    "source": "gliner"
                })
        except Exception as e:
            logger.error(f"GLiNER prediction failed: {e}")

    dict_ents = []
    if models.get("matcher") and models.get("medspacy"):
        nlp = models["medspacy"]
        doc = nlp(text)
        matches = models["matcher"](doc)
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            label, kg_type = string_id.split("||")
            span_doc = doc[start:end]
            dict_ents.append({
                "text": span_doc.text, "label": label, "kg_type": kg_type,
                "span": (span_doc.start_char, span_doc.end_char), "score": 1.0, "source": "expert_dictionary"
            })
            
    # Merge logic (ưu tiên dictionary)
    if not dict_ents: return sorted(gliner_ents, key=lambda x: x['span'][0])
    final_entities = list(dict_ents)
    dict_ranges = {i for d in dict_ents for i in range(d['span'][0], d['span'][1])}
    for g_ent in gliner_ents:
        is_overlapped = any(i in dict_ranges for i in range(g_ent['span'][0], g_ent['span'][1]))
        if not is_overlapped:
            final_entities.append(g_ent)
            
    return sorted(final_entities, key=lambda x: x['span'][0])

def run(state: MedCOTState) -> MedCOTState:
    logger.info("--- Running Step 1: Entity Extraction (Bulletproof Version) ---")
    models = load_models_bulletproof()
    
    # Dừng nếu không có model nào tải được
    if not models.get("gliner") and not models.get("matcher"):
        state.log("1_EXTRACTION", "CRITICAL_FAILURE", "No extraction models could be loaded.")
        logger.critical("CRITICAL: Both GLiNER and PhraseMatcher failed to load. Cannot proceed with extraction.")
        return state

    all_raw_mentions = []
    logger.info("  -> Extracting from query...")
    q_ments = _run_ner_on_text(state.normalized_query, models)
    for m in q_ments: m['source_doc'] = 'query'
    all_raw_mentions.extend(q_ments)

    if state.normalized_patient_context:
        logger.info("  -> Extracting from patient context...")
        c_ments = _run_ner_on_text(state.normalized_patient_context, models)
        for m in c_ments: m['source_doc'] = 'patient_context'
        all_raw_mentions.extend(c_ments)
    
    if not all_raw_mentions:
        state.log("1_EXTRACTION", "SKIPPED", "No entities found.")
        logger.warning("No entities found in any text.")
        return state

    for m in all_raw_mentions:
        if "kg_type" not in m:
            m["kg_type"] = config.INTERNAL_LABEL_TO_KG_TYPE_MAP.get(m["label"], "Phenotype")

    # Xử lý context (phủ định, etc.) nếu medspacy đã tải
    final_mentions = []
    if models.get("medspacy"):
        logger.info("  -> Running context analysis (negation, etc.)...")
        full_text = (state.normalized_query or "") + "\n" + (state.normalized_patient_context or "")
        nlp = models["medspacy"]
        doc = nlp(full_text)
        
        valid_spans = []
        offset = len(state.normalized_query or "") + 1
        for m in all_raw_mentions:
            start, end = m["span"]
            if m["source_doc"] == 'patient_context':
                start, end = start + offset, end + offset
            
            span = doc.char_span(start, end, label=m["label"])
            if span:
                span._.set("source_mention", m)
                valid_spans.append(span)

        doc.ents = filter_spans(valid_spans)
        for ent in doc.ents:
            src = ent._.get("source_mention")
            attrs = {'negated': ent._.is_negated, 'historical': ent._.is_historical, 'hypothetical': ent._.is_hypothetical}
            final_mentions.append(Mention(text=src["text"], label=src["label"], span=src["span"], score=src["score"], source=src["source_doc"], kg_type=src["kg_type"], attributes=attrs))
    else:
        logger.warning("MedSpaCy not loaded, skipping context analysis.")
        for m in all_raw_mentions:
             final_mentions.append(Mention(text=m["text"], label=m["label"], span=m["span"], score=m["score"], source=m["source_doc"], kg_type=m["kg_type"]))
            
    state.mentions = final_mentions
    state.log("1_EXTRACTION", "SUCCESS", metadata={"count": len(final_mentions)})
    logger.info(f"--- Step 1 Finished. Extracted {len(final_mentions)} mentions. ---")
    return state