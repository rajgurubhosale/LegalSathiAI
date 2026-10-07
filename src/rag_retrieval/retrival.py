
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import read_config_file
from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch

import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

class Retrieval:

    def __init__(self, top_n=10):
        try:      

            self.config = read_config_file()
            self.top_n  = top_n


            self.embedding_model = self._load_model()
            self.vectorstore = self._load_vector_db()
            
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_n}
            )
            self.bm25_retriever = self._load_bm25_retriever()

      
        except Exception as e:
            raise MyException(e,sys)        
    

    def warmup(self) -> None:
        try:
            logger.info("Warming up retrieval")
            self.retriever.invoke("warmup")      
            logger.info("Retrieval warm-up complete")
        except Exception as e:
            raise MyException(e, sys)

    def _load_model(self):

        return HuggingFaceEmbeddings(
            model_name=self.config["embedding"]["model_name"],
            encode_kwargs={"normalize_embeddings": True},
        )


    def _load_vector_db(self):
        db_config = self.config["vectorstore"]

        collection = Chroma(
            collection_name=db_config["db_name"],
            embedding_function=self.embedding_model,
            persist_directory=db_config["db_path"],
            create_collection_if_not_exists=False,

        )
        
        count = collection._collection.count()
        
        if count == 0:
            logger.error(
                f"Vector store '{db_config['db_name']}' "
                f"at '{db_config['db_path']}' is empty (0 vectors)"
            )
            raise MyException("Vector store is empty", sys)
        
        logger.info(f"Vector store loaded with {count} vectors")
        
        return collection
    
    
    def _load_bm25_retriever(self):
        raw = self.vectorstore.get(include=["documents", "metadatas"])

        docs = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(raw["documents"], raw["metadatas"])
            if text and text.strip()
        ]

        return BM25Retriever.from_documents(
            docs,
            k=self.top_n,
            preprocess_func=lambda text: re.findall(r"\w+", text.lower()),
    )

    def hybrid_retrieve_invoke(self, user_query, rrf_k=60):
        try:
            dense_docs = self.retrieve_invoke(user_query)
            sparse_docs = self.bm25_retriever.invoke(user_query)

            scores = {}
            documents = {}

            for results in (dense_docs, sparse_docs):
                seen = set()

                for rank, doc in enumerate(results, 1):
                    key = (
                        doc.metadata.get("act_name"),
                        doc.metadata.get("chunk_id") or doc.page_content,
                    )

                    if key in seen:
                        continue
                    seen.add(key)

                    scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank)
                    documents[key] = doc

            ranked_keys = sorted(scores, key=scores.get, reverse=True)

            return [documents[key] for key in ranked_keys[:self.top_n]]

        except Exception as e:
            raise MyException(e, sys)
        
    def retrieve_invoke(self,user_query):
        try: 
            return self.retriever.invoke(user_query)
        except Exception as e:
            raise MyException(e,sys)

        
class Reranker:
    
    def __init__(self,rerank_k:int):
        self.config = read_config_file()
        self.rerank_k = rerank_k
        self.reranker_model = self._load_reranker_model()
    
    def warmup(self) -> None:
        try:
            logger.info("Warming up reranker")
            self.reranker_model.client.predict([("warmup query", "warmup document")])
            logger.info("Reranker warm-up complete")
        except Exception as e:
            raise MyException(e, sys)

    def _load_reranker_model(self):
        return HuggingFaceCrossEncoder(
                model_name=self.config["rerank"]["model_name"],
                model_kwargs={
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
            )
    
    
    def rerank(self, docs: list, query: str, return_all: bool = False) -> list:
        if not docs:
            return []

        pairs = [
            (
                query,
                f"Act: {doc.metadata.get('act_name', 'Unknown')}\n{doc.page_content}",
            )
            for doc in docs
        ]

        predictions = self.reranker_model.client.predict(
            pairs,
            batch_size=16,
        )

        scored = sorted(
            zip(predictions, docs),
            key=lambda item: item[0],
            reverse=True,
        )

        return scored if return_all else scored[:self.rerank_k]
