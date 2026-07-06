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


    def embed_bns_first_section(self, chunk_path: str) -> None:
        """Embed BNS first section definition chunks."""
        # first section — flat list of strings
        logger.info(f"Embedding BNS first section: chunks")

        texts = self._load_chunks(chunk_path)

        embeddings = self.model.encode(texts, batch_size=32).tolist()

        self.collection.add(
            ids        = [f"BNS_DEF_{i}" for i in range(len(texts))],
            embeddings = embeddings,
            documents  = texts,
            metadatas  = [
                {
                    "parent_id": str(i),
                    "act":       "BNS",
                    "section":   "2",       
                    "type":      "definition"
                }
                for i in range(len(texts))
            ]
        )
        logger.info(f"Embedding BNS first section Completed")

    def embed_bns_second_section(self, chunk_path: str) -> None:
        """Embed BNS second section parent-child chunks."""
        try:
            chunks     = self._load_chunks(chunk_path)
           
            texts      = [c["child_text"] for c in chunks["children"]]
            embeddings = self.model.encode(texts, batch_size=32).tolist()

            self.collection.add(
            ids        = [f"{c['parent_id']}_c{i}" for i, c in enumerate(chunks["children"])],
            embeddings = embeddings,
            documents  = texts,
            metadatas  = [
                {
                    "parent_id":     c["parent_id"],
                    "act":           c["metadata"].get("act","BNS"),
                    "section":       c["metadata"].get("section",""),
                    "chapter":       c["metadata"].get("chapter",""),
                    "chapter_title": c["metadata"].get("chapter_title",""),
                    "section_title": c["metadata"].get("section_title",""),
                    "type":          "section"
                }
                for c in chunks["children"]
                ])
            
            logger.info(f"Embedded BNS second section: {len(chunks['children'])} chunks")
        except Exception as e:
            logger.error("Failed to embed BNS second section")
            raise MyException(e, sys)

    def embed_bnss_sections(self, chunk_path: str) -> None:
        """Embed BNSS sections parent-child chunks."""
        try:
            chunks     = self._load_chunks(chunk_path)
            children   = chunks["children"]   # ← flat list directly

            texts      = [c["child_text"] for c in children]
            embeddings = self.model.encode(texts, batch_size=32).tolist()

            self.collection.add(
                ids        = [f"{c['parent_id']}_c{i}" for i, c in enumerate(children)],
                embeddings = embeddings,
                documents  = texts,
                metadatas  = [
                    {
                        "parent_id":     c["parent_id"],
                        "act":           "BNSS",
                        "section":       c["metadata"].get("section_data",  ""),
                        "chapter":       c["metadata"].get("chapter",        ""),
                        "chapter_title": c["metadata"].get("chapter_title",  ""),
                        "type":          "section"
                    }
                    for c in children
                ]
            )
            logger.info(f"Embedded BNSS sections: {len(children)} chunks")

        except Exception as e:
            logger.error("Failed to embed BNSS sections")
            raise MyException(e, sys)
    
    def embed_bnss_tables(self, chunk_path: str) -> None:
        """Embed BNSS schedule 1 table chunks."""
        try:
            chunks     = self._load_chunks(chunk_path)  # flat list of dicts
            texts = [c["full_text"] for c in chunks]
            embeddings = self.model.encode(texts, batch_size=32).tolist()

            self.collection.add(
                ids        = [f"{c['parent_id']}_c{i}" for i, c in enumerate(chunks)],
                embeddings = embeddings,
                documents  = texts,
                metadatas  = [
                    {
                        "parent_id":  c["parent_id"],
                        "act":        c["metadata"].get("act",        "BNSS"),
                        "source_act": c["metadata"].get("source_act", "BNS"),
                        "section":    c["metadata"].get("section",    ""),
                        "type":       c["metadata"].get("type",       "schedule_1")
                    }
                    for c in chunks
                ]
            )
            logger.info(f"Embedded BNSS tables: {len(chunks)} chunks")

        except Exception as e:
            logger.error("Failed to embed BNSS tables")
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

            self.embedder.embed_bns_first_section(
                self.config['paths']['chunks']['bns']['first_section']
            )
            self.embedder.embed_bns_second_section(
                self.config['paths']['chunks']['bns']['second_section']
            )
            self.embedder.embed_bnss_sections(
                self.config['paths']['chunks']['bnss']['sections']
            )
            self.embedder.embed_bnss_tables(
                self.config['paths']['chunks']['bnss']['tables']
            )

            logger.info("Embedding pipeline completed")

        except Exception as e:
            logger.critical("Embedding pipeline failed")
            raise MyException(e, sys)


if __name__ == '__main__':
    runner = Pipeline()
    runner.run()