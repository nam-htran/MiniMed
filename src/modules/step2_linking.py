# Tệp: src/modules/step2_linking.py (PHIÊN BẢN FIX LỖI ID=NONE)
import logging
from src.core.state import MedCOTState, LinkedEntity, LinkedCandidate
from src.utils.neo4j_connect import db_connector
from src.utils.umls_normalizer import umls_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("step2_linking")

def _search_neo4j(text: str, kg_type: str = None):
    """Hàm tìm kiếm cốt lõi trong Neo4j (Case Insensitive)"""
    if not db_connector: return None
    
    # --- SỬA ĐỔI QUAN TRỌNG ---
    # Sử dụng hàm coalesce(n.id, elementId(n))
    # Ý nghĩa: Nếu n.id bị Null thì lấy elementId(n) (ID nội bộ của Neo4j, luôn tồn tại)
    query = """
    MATCH (n) 
    WHERE toLower(n.name) = toLower($text)
    RETURN 
        coalesce(n.id, elementId(n)) as node_id, 
        labels(n)[0] as node_label, 
        n.name as preferred_name
    LIMIT 1
    """
    try:
        res = db_connector.run_query(query, {"text": text})
        if res: return res[0]
    except Exception as e:
        logger.error(f"Error querying Neo4j: {e}")
        return None
    
    return None

def run(state: MedCOTState) -> MedCOTState:
    # Đảm bảo UMLS đã kết nối
    try:
        umls_service.connect()
    except Exception:
        logger.warning("UMLS service not available, skipping synonyms.")
    
    final_linked = []
    
    for mention in state.mentions:
        le = LinkedEntity(source_mention=mention)
        found_candidate = None
        method = "failed"

        # 1. Thử tìm trực tiếp (Direct Match)
        res = _search_neo4j(mention.text)
        if res:
            found_candidate = res
            method = "direct_exact"
        
        # 2. Nếu thất bại -> Dùng UMLS để mở rộng từ đồng nghĩa (Synonym Expansion)
        if not found_candidate:
            logger.info(f"🔍 Direct match failed for '{mention.text}'. Asking UMLS...")
            synonyms = []
            try:
                synonyms = umls_service.get_synonyms(mention.text)
            except Exception as e:
                logger.error(f"UMLS error: {e}")

            if synonyms:
                logger.info(f"   -> UMLS found synonyms: {synonyms[:3]} ...")
                # Thử từng synonym trong Neo4j
                for syn in synonyms:
                    res = _search_neo4j(syn)
                    if res:
                        found_candidate = res
                        method = f"umls_synonym ({syn})"
                        logger.info(f"   ✅ MATCHED via synonym: '{syn}' -> {res['preferred_name']}")
                        break
            else:
                logger.info("   -> UMLS found no synonyms.")

        # 3. Gán kết quả
        if found_candidate:
            # Đảm bảo node_id luôn là string (phòng hờ)
            safe_node_id = str(found_candidate["node_id"]) if found_candidate["node_id"] is not None else "UNKNOWN_ID"
            
            candidate = LinkedCandidate(
                node_id=safe_node_id, 
                node_label=found_candidate["node_label"],
                preferred_name=found_candidate["preferred_name"],
                score=1.0,
                source=method
            )
            le.link_status = "linked"
            le.best_candidate = candidate
            le.candidates = [candidate]
            logger.info(f"✅ Linked '{mention.text}' -> '{candidate.preferred_name}' (ID: {candidate.node_id})")
        else:
            logger.warning(f"❌ Could not link '{mention.text}' even with UMLS synonyms.")

        final_linked.append(le)

    state.linked_entities = final_linked
    
    # Cập nhật seed nodes cho các bước sau
    linked_ids = [le.best_candidate.node_id for le in final_linked if le.link_status == "linked"]
    
    if linked_ids:
        # Lấy elementId thực tế để query đồ thị (Bước 4)
        # Vì node_id bây giờ có thể là elementId hoặc id gốc, ta query lại để chắc chắn lấy elementId
        # Lưu ý: Nếu node_id đã là elementId thì query này vẫn chạy tốt nếu ta dùng WHERE elementId(n) = ... 
        # Nhưng để đơn giản và an toàn, ta dùng name để map lại elementId một lần nữa cho danh sách seed
        linked_names = [le.best_candidate.preferred_name for le in final_linked if le.link_status == "linked"]
        q = "MATCH (n) WHERE n.name IN $names RETURN elementId(n) as eid"
        r = db_connector.run_query(q, {"names": linked_names})
        state.seed_nodes = [x['eid'] for x in r]
    else:
        state.seed_nodes = []

    state.log("2_LINKING", "SUCCESS", {"count": len(state.seed_nodes)})
    return state