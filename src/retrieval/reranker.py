
from sentence_transformers import CrossEncoder
from src.logger import *
from src.exception  import *
from src.utils.main_utils import read_config_file
import json
import sys
from src.retrieval.parent_store import ParentStore   

class Reranker:
    
    def __init__(self,rerank_k:int = 3):

        try:
            self.config = read_config_file()    
            self.rerank_model = CrossEncoder(self.config['rerank']['cross_encoder_model_name'])        
            self.rerank_k = rerank_k
            self.parent_store = ParentStore()        
            
        except Exception as e:
            raise MyException(e,sys)
        
    
    def _load_parent_json(self):
        
        with open(self.config['paths']['parent_store_path'],'r',encoding='utf8') as f:
            parent_merge_json = json.load(f)
        
        return parent_merge_json
    
    def _make_pairs_data(self, docs: list, query: str) -> tuple:
        parent_ids = []
        child_chunks = []
        metadatas = []
        
        for d in docs:
            parent_ids.append(d.metadata.get("parent_id"))
            child_chunks.append(d.page_content.strip())
            metadatas.append(d.metadata)  
        
        pairs = [[query, c] for c in child_chunks]
        
        return pairs, parent_ids, child_chunks, metadatas
    
    def _get_top_chunks(self, docs, query):
        pairs, parent_ids, child_chunks, metadatas = self._make_pairs_data(docs, query)
        predictions = self.rerank_model.predict(pairs, batch_size=16)

        scored = sorted(zip(predictions, child_chunks, parent_ids, metadatas), reverse=True, key=lambda x: x[0])
        return scored[:self.rerank_k]
   
    
    
    def _build_header(self, metadata: dict, source_num: int) -> str:
        """
        Builds a citation header from whatever fields exist in metadata.
        Handles heterogeneous schemas (BNS sections, BNSS sections, BNSS tables)
        without hardcoding assumptions about which fields are present.
        """
        metadata = metadata or {}

        act = metadata.get("act", "")
        section = metadata.get("section", "")

        parts = [f"[Source {source_num}] {act} Section {section}".strip()]

        # Preferred, human-readable fields — added in priority order, only if present
        for field, label in [
            ("section_title", None),          # e.g. "Punishments."
            ("chapter_title", "Chapter"),      # e.g. "OF PUNISHMENTS"
            ("type", "Type"),                 # e.g. "schedule_1"
            ("source_act", "Source Act"),      # e.g. "BNS" (for BNSS schedule rows referencing BNS)
        ]:
            value = metadata.get(field)
            if value:
                parts.append(f"{label}: {value}" if label else value)

        # Catch-all: any other metadata fields not already used or explicitly suppressed —
        # future-proofs against schema fields you haven't anticipated
        suppressed = {"act", "section", "parent_id", "section_title", "chapter_title", "type", "source_act", "chapter", "section_data"}
        for key, value in metadata.items():
            if key not in suppressed and value:
                parts.append(f"{key}: {value}")

        return ", ".join(parts)


    def rerank_invoke(self, docs, query):
        top_chunks = self._get_top_chunks(docs, query)

        context_blocks = []
        for i, (score, child_text, parent_id, metadata) in enumerate(top_chunks):
            
            
            text = self.parent_store.get(parent_id) or child_text            
        

            header = self._build_header(metadata, source_num=i + 1)
            context_blocks.append(f"{header}\n{text}")

        return "\n\n".join(context_blocks)
        
    
  

 