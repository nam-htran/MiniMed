# src/core/state.py
from typing import List, Optional, Any, Dict, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Mention(BaseModel):
    text: str
    label: str
    span: Tuple[int, int]
    score: float
    source: str
    kg_type: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

class LinkedCandidate(BaseModel):
    node_id: str
    node_label: str
    preferred_name: str
    score: float
    source: str

class LinkedEntity(BaseModel):
    source_mention: Mention
    candidates: List[LinkedCandidate] = Field(default_factory=list)
    link_status: str = "unlinked"
    best_candidate: Optional[LinkedCandidate] = None

class MedCOTState(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)

    raw_query: str
    patient_context: Optional[str] = None
    normalized_query: Optional[str] = None
    normalized_patient_context: Optional[str] = None 
    sentences: List[Dict[str, Any]] = Field(default_factory=list)
    mentions: List[Mention] = Field(default_factory=list)
    linked_entities: List[LinkedEntity] = Field(default_factory=list)
    seed_nodes: List[str] = Field(default_factory=list)
    unlinked_mentions: List[Mention] = Field(default_factory=list)
    graph_refs: Dict[str, Any] = Field(default_factory=dict)
    gcot: Dict[str, Any] = Field(default_factory=dict)
    candidate_paths: List[Dict[str, Any]] = Field(default_factory=list)
    verified_path: List[Dict[str, Any]] = Field(default_factory=list)
    global_confidence: float = 0.0
    reasoning_mode: str = "Abstain"
    final_answer: Optional[str] = None
    safety_flags: List[Dict[str, Any]] = Field(default_factory=list)
    logs: List[Dict[str, Any]] = Field(default_factory=list)

    def log(self, step: str, status: str, message: Any = "", metadata: dict = None):
        if metadata is None: metadata = {}
        
        # --- FIX P0-4: Auto-detect metadata ---
        if isinstance(message, dict) and not metadata:
            metadata = message
            message_str = ""
        else:
            message_str = str(message)
            
        self.logs.append({
            "step": step,
            "status": status,
            "message": message_str,
            "timestamp": datetime.now().isoformat(),
            **metadata
        })