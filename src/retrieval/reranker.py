
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
        

    
    def _make_pairs_data(self, docs: list, query: str) -> tuple:
        parent_ids = []
        child_chunks = []
        metadatas = []
        scoring_texts = []

        for d in docs:
            parent_id = d.metadata.get("parent_id")
            child_text = d.page_content.strip()

            parent_ids.append(parent_id)
            child_chunks.append(child_text)
            metadatas.append(d.metadata)

            parent_text = self.parent_store.get(parent_id)
            scoring_texts.append(parent_text if parent_text else child_text)

        pairs = [[query, t] for t in scoring_texts]

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

        for field, label in [
            ("section_title", None),
            ("chapter_title", "Chapter"),
            ("source_act", "Source Act"),
        ]:
            value = metadata.get(field)
            if value:
                parts.append(f"{label}: {value}" if label else value)

        return ", ".join(parts)   # no catch-all loop after this

    def rerank_invoke(self, docs, query):
    
        top_chunks = self._get_top_chunks(docs, query)

        context_blocks = []
        for i, (score, child_text, parent_id, metadata) in enumerate(top_chunks):
            
            
            text = self.parent_store.get(parent_id) or child_text            

            header = self._build_header(metadata, source_num=i + 1)
            context_blocks.append(f"{header}\n{text}")

        output = "\n\n".join(context_blocks)
        
        
        return output,context_blocks
        
    
  

 