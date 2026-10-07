import json
import sys
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import read_config_file


class VectorStoreManager:
    """Handles creation and management of the Chroma vector store."""

    def __init__(self, collection_name: str, persist_directory: str, embedding_model_name: str):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )


    def add_documents(self, documents: List[Document]):
        self.store.add_documents(
            documents,
            ids=[doc.metadata["chunk_id"] for doc in documents],
        )

    def count(self) -> int:
        return self.store._collection.count()


def load_chunks(path: Path) -> List[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                docs.append(Document(page_content=record["page_content"],
                                     metadata=record["metadata"]))
    return docs


def main(batch_size: int = 128):
    try:
        config = read_config_file()
        chunks_path = Path(config["ingestion"]["chunks_path"])

        if not chunks_path.exists():
            raise FileNotFoundError(f"Run chunking first, file not found: {chunks_path}")

        docs = load_chunks(chunks_path)
        if not docs:
            raise ValueError("chunks.jsonl is empty")

        ids = [d.metadata["chunk_id"] for d in docs]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk_id values found in chunks.jsonl")

        manager = VectorStoreManager(
            collection_name=config["vectorstore"]["db_name"],
            persist_directory=config["vectorstore"]["db_path"],
            embedding_model_name=config["embedding"]["model_name"],
        )

        existing = manager.count()
        if existing > 0:
            raise ValueError(f"... already has {existing} vectors. Change db_path or db_name first.")
                
        for i in range(0, len(docs), batch_size):
            manager.add_documents(docs[i:i + batch_size])
            logger.info(f"Embedded {min(i + batch_size, len(docs))}/{len(docs)} chunks")

        if manager.count() != len(docs):
            raise ValueError(f"Expected {len(docs)} vectors, found {manager.count()}")

        logger.info(f"Embedding complete: {manager.count()} vectors stored")

    except Exception as e:
        logger.exception("Embedding failed")
        raise MyException(e, sys) from e


if __name__ == "__main__":
    main()