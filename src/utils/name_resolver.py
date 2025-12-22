# Tệp: src/utils/name_resolver.py (PHIÊN BẢN XỬ LÝ LIST AN TOÀN)
import requests
import logging
from typing import List

logger = logging.getLogger("NAME_RESOLVER")
SRI_LOOKUP_URL = "https://name-resolution-sri.renci.org/lookup"

class NameResolver:
    def __init__(self):
        self.cache = {}

    def resolve_names_to_curies(self, names: List[str]) -> List[str]:
        unique_names = list(set([n.strip() for n in names if n.strip()]))
        resolved_curies = []

        for name in unique_names:
            if name.lower() in self.cache:
                val = self.cache[name.lower()]
                if val: resolved_curies.append(val)
                continue

            try:
                params = {"string": name, "limit": 1}
                response = requests.post(SRI_LOOKUP_URL, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    best_curie = None

                    # LOGIC XỬ LÝ LIST/DICT LINH HOẠT
                    if isinstance(data, dict) and data:
                        # {"CURIE": "Label"}
                        best_curie = list(data.keys())[0]
                    elif isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict):
                            # [{"curie": "...", ...}] HOẶC [{"CURIE": "Label"}]
                            best_curie = first.get('curie') or first.get('id')
                            if not best_curie:
                                best_curie = list(first.keys())[0]
                        elif isinstance(first, str):
                            best_curie = first

                    if best_curie:
                        self.cache[name.lower()] = best_curie
                        resolved_curies.append(best_curie)
                        logger.info(f"✅ SRI Resolved '{name}' -> {best_curie}")
                    else:
                        logger.warning(f"❌ No ID found for name: '{name}' (Response: {data})")
                        self.cache[name.lower()] = None
                else:
                    self.cache[name.lower()] = None
            except Exception as e:
                logger.error(f"Name Resolver failed for '{name}': {e}")
                self.cache[name.lower()] = None

        return list(set(resolved_curies))

name_resolver = NameResolver()