# Tệp: src/modules/step4_retrieval.py (PHIÊN BẢN FIX KẾT NỐI ID)
import logging
import uuid
from typing import Dict, Any, List
from src.core.state import MedCOTState
from src.utils.neo4j_connect import db_connector
from src.utils.arax_client import arax_client
from src.utils.name_resolver import name_resolver

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("step4_retrieval")

def _run_simple_expansion(seed_ids: List[str]) -> Dict[str, Any]:
    if not seed_ids or db_connector is None: return {"nodes": [], "edges": []}
    logger.info(f"   🕸 [Simple Expansion] Getting seed nodes and direct neighbors...")
    
    # Dùng coalesce để đảm bảo luôn lấy được ID hợp lệ (ưu tiên id gốc, fallback sang elementId)
    query = """
    MATCH (seed) WHERE elementId(seed) IN $seeds OR seed.id IN $seeds
    OPTIONAL MATCH (seed)-[r]-(neighbor)
    WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) as all_nodes_list, collect(DISTINCT r) as all_rels_list
    RETURN [node in all_nodes_list WHERE node IS NOT NULL | 
            {id: coalesce(node.id, elementId(node)), labels: labels(node), name: node.name, provenance: 'PrimeKG', element_id: elementId(node)}] as nodes,
           [rel in all_rels_list WHERE rel IS NOT NULL | 
            {id: elementId(rel), source: coalesce(startNode(rel).id, elementId(startNode(rel))), target: coalesce(endNode(rel).id, elementId(endNode(rel))), type: type(rel), provenance: 'PrimeKG'}] as relationships
    """
    try:
        results = db_connector.run_query(query, {"seeds": seed_ids})
        if not results: return {"nodes": [], "edges": []}
        record = results[0]
        return {"nodes": record.get("nodes", []), "edges": record.get("relationships", [])}
    except Exception as e:
        logger.error(f"Simple Expansion Query on PrimeKG failed: {e}")
        return {"nodes": [], "edges": []}

def _build_patient_state_graph(state: MedCOTState) -> Dict[str, Any]:
    psg_nodes, psg_edges, pid = [], [], f"PATIENT_{state.query_id[:8]}"
    psg_nodes.append({"id": pid, "label": "Patient", "name": "The Patient", "provenance": "PSG"})
    for le in state.linked_entities:
        if le.link_status == "linked" and le.source_mention.source == "patient_context":
            evt_id = f"EVT_{uuid.uuid4().hex[:6]}"
            # Link vào node_id đã được chuẩn hóa ở Step 2
            target_id = str(le.best_candidate.node_id)
            psg_nodes.append({"id": evt_id, "label": "Observation", "name": le.source_mention.text, "provenance": "PSG"})
            psg_edges.extend([
                {"source": evt_id, "target": target_id, "type": "GROUNDED_IN", "provenance": "PSG"}, 
                {"source": pid, "target": evt_id, "type": "PRESENTS_WITH", "provenance": "PSG"}
            ])
    return {"nodes": psg_nodes, "edges": psg_edges}

def run(state: MedCOTState, use_arax_fallback: bool = True) -> MedCOTState:
    logger.info("🚀 Step 4: Hybrid Retrieval (Local + SRI Name Resolution)")
    
    local_graph = _run_simple_expansion(state.seed_nodes)
    arax_graph = {"nodes": [], "edges": []}
    
    # --- LOGIC QUAN TRỌNG: Lấy dữ liệu từ ARAX và cập nhật Seed Nodes ---
    if use_arax_fallback and state.mentions:
        entity_names = list(set([m.text for m in state.mentions]))
        logger.info(f"Resolving names externally: {entity_names}")
        
        resolved_curies = name_resolver.resolve_names_to_curies(entity_names)
        logger.info(f"Got resolved CURIEs: {resolved_curies}")
        
        # [FIX] Thêm CURIEs từ ARAX vào seed_nodes để Step 6 có thể tìm đường đi
        if resolved_curies:
            state.seed_nodes.extend(resolved_curies)
            state.seed_nodes = list(set(state.seed_nodes)) # Remove duplicates
            logger.info(f"✅ Updated Seed Nodes with External IDs: {resolved_curies}")
        
        if len(resolved_curies) >= 2:
            arax_edges = arax_client.query_kg2(resolved_curies, max_results=10) # Tăng max results
            if arax_edges:
                arax_nodes_map = {}
                for edge in arax_edges:
                    s_id, s_name = edge.get('source_id', edge.get('source')), edge.get('source')
                    t_id, t_name = edge.get('target_id', edge.get('target')), edge.get('target')
                    
                    # Đảm bảo ID node trùng với ID trong seed_nodes (là CURIE)
                    arax_nodes_map[s_id] = {"id": s_id, "label": "ExternalEntity", "name": s_name, "provenance": "ARAX/KG2"}
                    arax_nodes_map[t_id] = {"id": t_id, "label": "ExternalEntity", "name": t_name, "provenance": "ARAX/KG2"}
                    
                    edge['source'], edge['target'] = s_id, t_id
                    
                arax_graph["nodes"], arax_graph["edges"] = list(arax_nodes_map.values()), arax_edges
                logger.info(f"✅ ARAX returned {len(arax_graph['nodes'])} nodes and {len(arax_graph['edges'])} edges.")

    psg = _build_patient_state_graph(state)
    logger.info("   - Merging graphs from all sources...")
    
    merged_nodes = local_graph.get("nodes", []) + arax_graph.get("nodes", []) + psg.get("nodes", [])
    merged_edges = local_graph.get("edges", []) + arax_graph.get("edges", []) + psg.get("edges", [])
    
    # Khử trùng lặp node theo ID
    final_node_map = {}
    for n in merged_nodes:
        if n.get('id'):
            final_node_map[n['id']] = n
            
    valid_node_ids = set(final_node_map.keys())
    
    # Chỉ giữ lại cạnh nào mà cả 2 đầu đều tồn tại trong danh sách node
    final_edges = [e for e in merged_edges if e.get('source') in valid_node_ids and e.get('target') in valid_node_ids]
    
    state.graph_refs["ckg_subgraph"] = {"nodes": list(final_node_map.values()), "edges": final_edges}
    
    arax_was_used = use_arax_fallback and bool(arax_graph.get("edges"))
    state.log("4_RETRIEVAL", "SUCCESS", metadata={"nodes": len(final_node_map), "edges": len(final_edges), "arax_used": arax_was_used})
    
    return state