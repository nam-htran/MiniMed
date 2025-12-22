# Tệp: src/utils/arax_client.py (PHIÊN BẢN RETRY + TỐI ƯU)
import requests
import logging
import json
import time
import hashlib
from pathlib import Path
from itertools import combinations

logger = logging.getLogger("ARAX_CLIENT")
ARAX_BASE_URL = "https://arax.ncats.io/api/arax/v1.4"

CACHE_DIR = Path(".cache/arax_queries")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRATION_SECONDS = 86400 * 7

class ARAXClient:
    def __init__(self):
        self.headers = {"accept": "application/json", "Content-Type": "application/json"}
        self.curie_cache = {}

    def resolve_names_to_curies(self, names: list) -> list:
        # (Giữ nguyên logic resolve name vì đã hoạt động tốt)
        unique_names = list(set([n for n in names if n and n not in self.curie_cache]))
        if unique_names:
            try:
                params = [('q', name) for name in unique_names]
                response = requests.get(f"{ARAX_BASE_URL}/entity", params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        for item in data:
                            label = item.get('label') or item.get('name')
                            curie = item.get('id')
                            if label and curie:
                                self._cache_result(label, curie)
                                for input_name in unique_names:
                                    if input_name.lower() == label.lower():
                                        self._cache_result(input_name, curie)
                    elif isinstance(data, dict):
                        for key, val in data.items():
                            self._cache_result(key, val)
            except Exception as e:
                logger.error(f"Error resolving names via ARAX: {e}")

        results = []
        for name in names:
            curie = self.curie_cache.get(name) or self.curie_cache.get(name.lower())
            if curie: results.append(curie)
        return list(set(results))

    def _cache_result(self, name, result):
        curie = None
        if isinstance(result, str): curie = result
        elif isinstance(result, dict): curie = result.get('identifier') or result.get('id') or result.get('curie')
        if curie and isinstance(curie, str):
            self.curie_cache[name] = curie
            self.curie_cache[name.lower()] = curie

    def _get_cache_key(self, identifiers):
        key_str = "arax_v1.4_optimized_" + "_".join(sorted([str(i).lower() for i in identifiers]))
        return hashlib.md5(key_str.encode()).hexdigest()

    def query_kg2(self, identifiers: list, max_results=5):
        if len(identifiers) < 2: return []
            
        cache_key = self._get_cache_key(identifiers)
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_EXPIRATION_SECONDS:
            logger.info(f"⚡ [Cache Hit] Loading ARAX connecting path for {identifiers}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f: return json.load(f)
            except Exception: pass

        logger.info(f"🌍 Querying ARAX (v1.4) for interactions between: {identifiers}...")
        
        id_pairs = list(combinations(identifiers, 2))
        all_results = []
        
        for id1, id2 in id_pairs:
            # --- TỐI ƯU HÓA QUERY ---
            # Chỉ tìm các cạnh có ý nghĩa tương tác thuốc để giảm tải server
            payload = {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": [id1]},
                            "n1": {"ids": [id2]}
                        },
                        "edges": {
                            "e0": {
                                "subject": "n0", 
                                "object": "n1",
                                # Chỉ tìm tương tác (interacts_with) hoặc liên quan (related_to)
                                # Điều này giúp query chạy NHANH HƠN và ít bị lỗi 503
                                "predicates": [
                                    "biolink:interacts_with",
                                    "biolink:affects",
                                    "biolink:related_to"
                                ]
                            }
                        }
                    }
                },
                "max_results": max_results,
                "submitter": "MedCOT_Agent"
            }

            # --- CƠ CHẾ RETRY (THỬ LẠI) ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(f"{ARAX_BASE_URL}/query", headers=self.headers, json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "message" in data and "knowledge_graph" in data["message"]:
                            kg = data["message"]["knowledge_graph"]
                            if kg:
                                parsed_edges = self._parse_trapi_to_medcot(kg.get("nodes", {}), kg.get("edges", {}))
                                all_results.extend(parsed_edges)
                                logger.info(f"  -> Found {len(parsed_edges)} interaction edges between '{id1}' and '{id2}'.")
                            else:
                                logger.info(f"  -> ARAX returned 0 paths (Graph empty) for '{id1}' and '{id2}'.")
                        break # Thành công thì thoát vòng lặp retry
                    
                    elif response.status_code == 503:
                        logger.warning(f"⚠️ ARAX Server busy (503). Retrying {attempt+1}/{max_retries} in 5s...")
                        time.sleep(5) # Đợi 5 giây rồi thử lại
                    else:
                        logger.warning(f"ARAX query failed {response.status_code}: {response.text[:200]}")
                        break # Lỗi khác 503 thì không thử lại

                except Exception as e:
                    logger.error(f"❌ Exception querying ARAX: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        logger.error("❌ Max retries exceeded.")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f: json.dump(all_results, f, ensure_ascii=False)
        except Exception: pass
            
        return all_results

    def _parse_trapi_to_medcot(self, nodes, edges):
        if not nodes or not edges: return []
        
        medcot_edges, node_map = [], {nid: n_info.get("name", nid) for nid, n_info in nodes.items()}
        for eid, e_info in edges.items():
            predicate = e_info.get("predicate", "related_to")
            if ":" in predicate: predicate = predicate.split(":")[-1]
            predicate = predicate.upper()

            primary_source = "ARAX/KG2"
            if "attributes" in e_info:
                for attr in e_info["attributes"]:
                    if attr.get("attribute_type_id") == "biolink:primary_knowledge_source":
                        primary_source = attr.get("value")
                        break

            medcot_edges.append({
                "source": node_map.get(e_info.get("subject"), e_info.get("subject")), 
                "target": node_map.get(e_info.get("object"), e_info.get("object")), 
                "type": predicate,
                "source_id": e_info.get("subject"), 
                "target_id": e_info.get("object"),
                "provenance": "ARAX/KG2", 
                "remote_source": primary_source
            })
        return medcot_edges

arax_client = ARAXClient()