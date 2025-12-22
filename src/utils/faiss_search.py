# src/utils/faiss_search.py
import logging
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logger = logging.getLogger("FAISS_SEARCH")

class FaissSearch:
    def __init__(self, index_dir: str = "data/kg_index", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "kg_faiss.index"
        self.meta_path = self.index_dir / "kg_nodes_meta.json"
        self.model_name = model_name
        
        self.index = None
        self.meta = None
        self.encoder = None
        self._load_resources()

    def _load_resources(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            msg = f"FAISS index not found at {self.index_dir}. Please run 'scripts/2_build_faiss.py'."
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        logger.info("Loading FAISS resources...")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.encoder = SentenceTransformer(self.model_name)
        logger.info(f"✅ FAISS resources loaded ({self.index.ntotal} vectors).")

    def search(self, query_text: str, k: int = 5) -> list[dict]:
        """Searches the index and returns top-k metadata."""
        if self.index is None:
            raise RuntimeError("Index is not loaded.")
        
        query_vector = self.encoder.encode([query_text], normalize_embeddings=True)
        distances, indices = self.index.search(np.asarray(query_vector, dtype="float32"), k)
        
        results = []
        for i in indices[0]:
            if i != -1: # FAISS returns -1 for empty slots
                results.append(self.meta[i])
        return results

# Singleton instance for easy import
try:
    faiss_retriever = FaissSearch()
except FileNotFoundError:
    faiss_retriever = None