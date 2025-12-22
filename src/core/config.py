# Tệp: src/core/config.py
LINKING_THRESHOLD = 0.7 
# (Các hằng số khác giữ nguyên như trước)
SPACY_MODEL_NAME = "xx_sent_ud_sm"
EXTRACTION_MODEL_NAME = "urchade/gliner_multi-v2.1"
ENTITY_LABELS = ["medical condition or disease", "symptom or sign", "drug or medication", "laboratory test", "anatomy or body part", "medical procedure", "gene or protein"]
GLINER_TO_INTERNAL_LABEL_MAP = {"medical condition or disease": "disease", "symptom or sign": "symptom", "drug or medication": "drug", "laboratory test": "lab_test", "medical procedure": "procedure", "anatomy or body part": "anatomy", "gene or protein": "gene"}
INTERNAL_LABEL_TO_KG_TYPE_MAP = {"disease": "Disease", "drug": "Drug", "symptom": "Effect/Phenotype", "anatomy": "Anatomy", "procedure": "Procedure", "lab_test": "Gene/Protein", "gene": "Gene/Protein"}
KG_TYPE_TO_INTERNAL_MAP = {v: k for k, v in INTERNAL_LABEL_TO_KG_TYPE_MAP.items()}
DEFAULT_EXTRACTION_THRESHOLD = 0.35
KNOWN_ENTITIES = {"diabetes": ("disease", "Disease"), "metformin": ("drug", "Drug"), "hypertension": ("disease", "Disease"), "warfarin": ("drug", "Drug"), "aspirin": ("drug", "Drug"), "kidney disease": ("disease", "Disease")}
DENSE_RETRIEVAL_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
KG_TYPE_TO_UMLS_STY_MAP = {"disease": ["Disease or Syndrome", "Neoplastic Process", "Pathologic Function", "Congenital Abnormality", "Mental or Behavioral Dysfunction", "Injury or Poisoning"], "symptom": ["Sign or Symptom", "Finding", "Laboratory or Test Result"], "drug": ["Pharmacologic Substance", "Clinical Drug", "Antibiotic", "Biologically Active Substance"], "procedure": ["Therapeutic or Preventive Procedure", "Diagnostic Procedure", "Health Care Activity"], "anatomy": ["Body Part, Organ, or Organ Component", "Anatomical Structure", "Body Location or Region", "Tissue"], "gene": ["Gene or Genome", "Amino Acid, Peptide, or Protein", "Enzyme"], "lab_test": ["Laboratory Procedure", "Diagnostic Procedure"]}
NODE_EMBEDDING_DIM = 384  
HGT_HIDDEN_CHANNELS = 128
HGT_NUM_HEADS = 4
NLI_MODEL_NAME = "cross-encoder/nli-distilroberta-base"
WEIGHTS = {"in_kg": 0.35, "link_pred": 0.05, "nli": 0.15, "causality": 0.15, "gcot": 0.10, "trust": 0.20}