import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from pathlib import Path
from src.core.state import MedCOTState
from src.models.dual_tower_gnn import CoGCoT_DualTower_GNN

# --- CẤU HÌNH LOGGING ĐỂ TẮT RÁC ---
# Tắt log DEBUG của PyRuSH và các thư viện khác để log gọn gàng
logger = logging.getLogger("step5_dual_tower")
# ------------------------------------

_encoder = None
GNN_MODEL_PATH = Path("models/gnn_dual_tower_weights.pth")

def load_encoder():
    global _encoder
    if _encoder is None: 
        logger.info("loading sentence transformer...")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2") 
    return _encoder

def _prepare_hetero_data_robust(nodes, edges, encoder):
    """
    Hàm này tạo data trên CPU, ta sẽ chuyển lên GPU sau.
    """
    data = HeteroData()
    if not nodes: return data, {}

    # Group nodes and create embeddings (trên CPU)
    grouped_nodes = {}
    node_id_to_idx = {} 
    
    for n in nodes:
        lbl = n.get("label", "Unknown").replace("/", "_").replace(" ", "_")
        if lbl not in grouped_nodes: grouped_nodes[lbl] = []
        current_idx = len(grouped_nodes[lbl])
        node_id_to_idx[n['id']] = (lbl, current_idx)
        grouped_nodes[lbl].append(n)

    for lbl, nlist in grouped_nodes.items():
        texts = [n.get("name", "Unknown") for n in nlist]
        embs = encoder.encode(texts, show_progress_bar=False)
        data[lbl].x = torch.tensor(embs, dtype=torch.float32)

    # Process Edges
    edge_index_map = {}
    for e in edges:
        s_id, t_id = e["source"], e["target"]
        if s_id not in node_id_to_idx or t_id not in node_id_to_idx:
            continue
        s_lbl, s_idx = node_id_to_idx[s_id]
        t_lbl, t_idx = node_id_to_idx[t_id]
        e_type = e.get("type", "RELATED").upper()
        triplet = (s_lbl, e_type, t_lbl)
        if triplet not in edge_index_map:
            edge_index_map[triplet] = [[], []]
        edge_index_map[triplet][0].append(s_idx)
        edge_index_map[triplet][1].append(t_idx)

    for triplet, indices in edge_index_map.items():
        if len(indices[0]) > 0:
            data[triplet].edge_index = torch.tensor(indices, dtype=torch.long)
    
    legacy_node_map = {}
    for lbl, nlist in grouped_nodes.items():
        legacy_node_map[lbl] = {n['id']: i for i, n in enumerate(nlist)}

    return data, legacy_node_map

def run(state: MedCOTState, num_think_steps: int = 2) -> MedCOTState:
    ug = state.graph_refs.get("ckg_subgraph")
    if not ug or not ug.get("nodes"):
        state.log("5_REASONING", "SKIPPED", "No subgraph")
        return state

    encoder = load_encoder()
    
    # --- SỬA LỖI DEVICE ---
    # 1. Xác định thiết bị đích (GPU nếu có, không thì CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"GNN running on device: {device}")
    
    # 2. Tạo Tensors (mặc định trên CPU hoặc GPU)
    q_emb = encoder.encode(state.normalized_query, convert_to_tensor=True) if state.normalized_query else torch.zeros(384)

    ckg_nodes = [n for n in ug["nodes"] if n.get("provenance") != "PSG"]
    psg_nodes = [n for n in ug["nodes"] if n.get("provenance") == "PSG"]
    ckg_edges = [e for e in ug["edges"] if e.get("provenance") != "PSG"]
    psg_edges = [e for e in ug["edges"] if e.get("provenance") == "PSG"]

    ckg_d, ckg_m = _prepare_hetero_data_robust(ckg_nodes, ckg_edges, encoder)
    psg_d, psg_m = _prepare_hetero_data_robust(psg_nodes, psg_edges, encoder)

    # 3. Chuyển tất cả mọi thứ lên cùng một device
    q_emb = q_emb.to(device)
    ckg_d = ckg_d.to(device)
    psg_d = psg_d.to(device)
    # -----------------------

    state.graph_refs["node_map"] = ckg_m

    if not ckg_d.node_types: 
        state.log("5_REASONING", "SKIPPED", "Empty CKG Data")
        return state

    try:
        # Load model và chuyển nó lên device
        model = CoGCoT_DualTower_GNN(ckg_d.metadata(), psg_d.metadata(), 128, 384, num_think_steps)
        model.to(device) # <--- QUAN TRỌNG
        
        if GNN_MODEL_PATH.exists(): 
            try:
                # Load weights vào model đã ở trên GPU
                model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device), strict=False)
            except Exception as load_err:
                logger.warning(f"Could not load GNN weights: {load_err}")
        
        model.eval()
        with torch.no_grad():
            # Bây giờ tất cả input và model đều ở trên GPU
            final_x, thoughts = model(ckg_d, psg_d, q_emb)

        # Chuyển kết quả về lại CPU để lưu trữ (numpy/json không đọc được tensor GPU)
        final_node_embeddings = {}
        for nt, feat in final_x.items():
            final_node_embeddings[nt] = feat.cpu().numpy()
        state.graph_refs["final_node_embeddings"] = final_node_embeddings
            
        state.gcot["thought_vectors"] = thoughts
        state.log("5_REASONING", "SUCCESS", {"thoughts": len(thoughts)})

    except Exception as e:
        logger.error(f"GNN Error handled gracefully: {e}", exc_info=True) # exc_info=True để in traceback
        state.log("5_REASONING", "FAILED_BUT_CONTINUED", str(e))

    return state