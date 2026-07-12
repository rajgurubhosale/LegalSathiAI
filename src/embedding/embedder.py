from sentence_transformers import SentenceTransformer
from src.exception import *
from src.logger import *
from src.utils.main_utils import read_config_file,ensure_path
import chromadb
import json
from dotenv import load_dotenv
import os 
from huggingface_hub import login

load_dotenv()
login(token=os.getenv('HF_TOKEN'))

class Embedder:
    """Embeds legal chunks and stores in ChromaDB."""

    def __init__(self, model_name: str, db_path: str):
        self.model      = self._load_model(model_name)
        self.collection = self._load_db(db_path)

    def _load_model(self, model_name: str):
        """Load sentence transformer embedding model."""
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
            return model
        except Exception as e:
            raise MyException(e, sys)
        

    def _load_db(self, db_path: str):
        """Connect to ChromaDB and get collection."""
        try:
            client     = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection("legalsaathi")
            logger.info(f"ChromaDB connected: {db_path}")
            return collection
        except Exception as e:
            raise MyException(e, sys)

    def _load_chunks(self, chunk_path: str) -> list:
        """Load chunks from JSON file."""
        with open(chunk_path, encoding="utf-8") as f:
            return json.load(f)

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        ChromaDB metadata values must be str/int/float/bool — None is NOT allowed
        and will raise an error. This replaces None with empty string, and handles
        the case where the whole metadata dict itself is None (e.g. BNS definitions).
        """
        if not metadata:
            return {}
        return {k: (v if v is not None else "") for k, v in metadata.items()}
        
    def embed_chunks(self, chunk_path: str) -> None:
        """Embed BNS second section parent-child chunks."""
        try:
            
            chunks     = self._load_chunks(chunk_path)
            
            children_data =chunks['children_data']
            ids = []
            texts = []
            metadatas = []
            
            for key,entry in children_data.items():
                
                parent_id = key
                metadata = self._sanitize_metadata(entry.get("metadata"))
                child_list = entry['children']
                
                for idx,child_text in enumerate(child_list):
                    child_text = child_text.strip()
                    if not child_text:
                        continue
                    
                    ids.append(f"{parent_id}_c{idx}")
                    texts.append(child_text)
                    metadatas.append(metadata)  


            if not texts:
                logger.warning(f"No child texts found for {chunk_path}")
                return
            
            embeddings = self.model.encode(texts, batch_size=32).tolist()

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Embedded: {chunk_path}, chunks: {len(texts)} ")
        except Exception as e:
            logger.error("Failed to embed BNS second section")
            raise MyException(e, sys)


class Pipeline:

    def __init__(self):
        self.config  = read_config_file()
        
        ensure_path(self.config['embedding']['db_path'])
        
        self.embedder = Embedder(
            model_name = self.config['embedding']['model_name'],
            
            db_path    = self.config['embedding']['db_path']
        )
    
    def run(self) -> None:
        try:
            logger.info("Embedding pipeline started")
 
            # One function, four sources — same call signature every time
            self.embedder.embed_chunks(
                self.config['paths']['chunks']['bns']['first_section'],
            )
            self.embedder.embed_chunks(
                self.config['paths']['chunks']['bns']['second_section'],
            )
            self.embedder.embed_chunks(
                self.config['paths']['chunks']['bnss']['sections'],
            )
            self.embedder.embed_chunks(
                self.config['paths']['chunks']['bnss']['tables'],
            )
 
            logger.info("Embedding pipeline completed")
 
        except Exception as e:
            logger.critical("Embedding pipeline failed")
            raise MyException(e, sys)
 
 
if __name__ == '__main__':
    runner = Pipeline()
    runner.run()