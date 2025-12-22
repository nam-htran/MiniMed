# src/modules/step6_path_generation.py
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from src.core.state import MedCOTState

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("step6_constrained_path_gen")
_models = {}

# --- NÂNG CẤP THEO CHỐT 6: RÀNG BUỘC TÌM KIẾM THEO NGỮ NGHĨA ---
SEMANTIC_CONSTRAINTS = {
    "TREATMENT": ["indication", "treats", "prevents", "mitigates"],
    "SAFETY": ["contraindication", "side effect", "adverse reaction", "risk_of", "causes", "interacts_with"],
    "DIAGNOSIS": ["biomarker", "associated_with", "has_symptom", "presents_with"],
    "GENERIC": [] # Chế độ Fallback không có ràng buộc
}

def load_models():
    if not _models:
        _models["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
        _models["reranker"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _models["embedder"], _models["reranker"]

def detect_query_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["treat", "cure", "therapy", "manage", "medication", "drug for"]): return "TREATMENT"
    if any(w in q for w in ["safe", "risk", "contraindicat", "bad", "interaction", "warn", "avoid"]): return "SAFETY"
    if any(w in q for w in ["diagnos", "test", "check", "symptom", "sign", "cause"]): return "DIAGNOSIS"
    return "GENERIC"

class ConstrainedPathGenerator:
    def __init__(self, state: MedCOTState, embedder):
        self.state = state
        self.embedder = embedder
        self.query_emb = embedder.encode(state.normalized_query, convert_to_tensor=True) if state.normalized_query else None
        self.intent = detect_query_intent(state.normalized_query)
        self.meta = {n['id']: n for n in state.graph_refs.get("ckg_subgraph", {}).get("nodes", [])}
        self.adj = self._build_adj(strict_mode=True)
        self.used_fallback = False

    def _build_adj(self, strict_mode=True):
        adj, edges = {}, self.state.graph_refs.get("ckg_subgraph", {}).get("edges", [])
        allowed = SEMANTIC_CONSTRAINTS.get(self.intent, []) if strict_mode else []
        
        valid_count = 0
        for e in edges:
            s, t, raw_type = e["source"], e["target"], e["type"].lower()
            if strict_mode and self.intent != "GENERIC" and allowed and not any(valid in raw_type for valid in allowed): continue
            valid_count += 1
            info = {"node": t, "edge_raw": e["type"], "edge_text": raw_type.replace("_", " "), "provenance": e.get("provenance", "DEFAULT")}
            adj.setdefault(s, []).append(info)
        
        logger.info(f"🕸 Adj built (Strict={strict_mode}, Intent={self.intent}). Valid edges: {valid_count}/{len(edges)}")
        return adj

    def enable_fallback(self):
        logger.warning("⚠️ No paths found with strict constraints. Switching to GENERIC mode.")
        self.intent = "GENERIC"
        self.adj = self._build_adj(strict_mode=False)
        self.used_fallback = True

    def search(self, width=50, depth=3):
        if self.query_emb is None or not self.state.seed_nodes: return []
        seeds = [s for s in self.state.seed_nodes if s in self.adj]
        if not seeds: return []
        
        beam = [(0.0, [{"node_id": s}]) for s in seeds]
        final = []
        
        for _ in range(depth): # Max path length = depth
            candidates = []
            for score, path in beam:
                curr_node_id = path[-1]['node_id']
                if len(path) > 1: final.append((score, path))
                
                neighbors = self.adj.get(curr_node_id, [])
                current_path_nodes = {step['node_id'] for step in path}
                valid_neighbors = [n for n in neighbors if n['node'] not in current_path_nodes] # Cycle Control
                if not valid_neighbors: continue
                
                texts = [f"{n['edge_text']} {self.meta.get(n['node'], {}).get('name', '')}" for n in valid_neighbors]
                sem_sims = util.cos_sim(self.query_emb, self.embedder.encode(texts, convert_to_tensor=True))[0].cpu().numpy()
                
                for i, nb in enumerate(valid_neighbors):
                    new_step = {"node_id": nb['node'], "edge_raw": nb["edge_raw"], "edge_text": nb["edge_text"], "provenance": nb["provenance"]}
                    candidates.append((score + float(sem_sims[i]), path + [new_step]))
            
            if not candidates: break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:width]
        
        results, seen_paths = [], set()
        for score, path in sorted(final + beam, key=lambda x: x[0], reverse=True):
            if len(path) < 2: continue
            clean_path, parts = [], []
            for i in range(len(path) - 1):
                s_name = self.meta.get(path[i]['node_id'], {}).get('name', 'Unknown')
                t_name = self.meta.get(path[i+1]['node_id'], {}).get('name', 'Unknown')
                step_info = {"source": path[i]['node_id'], "target": path[i+1]['node_id'], "edge": path[i+1]['edge_raw'], "edge_text": path[i+1]['edge_text'], "provenance": path[i+1]['provenance']}
                clean_path.append(step_info)
                parts.append(f"{s_name} --[{step_info['edge_text']}]--> {t_name}")
            
            text_repr = " ".join(parts)
            if text_repr not in seen_paths:
                seen_paths.add(text_repr)
                results.append({"path": clean_path, "text_repr": text_repr, "score": float(score)})
        return results[:width]

def run(state: MedCOTState, beam_width: int = 50, max_path_length: int = 3) -> MedCOTState: # Max hops = 3-1 = 2
    try:
        embedder, reranker = load_models()
        gen = ConstrainedPathGenerator(state, embedder)
        
        paths = gen.search(width=beam_width, depth=max_path_length)
        if not paths:
            gen.enable_fallback()
            paths = gen.search(width=beam_width, depth=max_path_length)
        
        if not paths: 
            state.log("6_PATH_GEN", "SKIPPED", {"msg": "No paths found even with fallback"})
            return state
        
        path_texts = [[state.normalized_query, p["text_repr"]] for p in paths]
        scores = reranker.predict(path_texts)
        for i, p in enumerate(paths): 
            p['final_score'] = 0.3 * p['score'] + 0.7 * (1 / (1 + np.exp(-scores[i])))
        
        state.candidate_paths = sorted(paths, key=lambda x: x['final_score'], reverse=True)[:10]
        state.log("6_PATH_GEN", "SUCCESS", {"count": len(state.candidate_paths), "intent": gen.intent, "fallback_used": gen.used_fallback})
        
    except Exception as e:
        logger.exception("Path Gen Error")
        state.log("6_PATH_GEN", "FAILED", str(e))
    return state