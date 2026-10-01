
from src.data_ingestion.ingest_and_chunk import *
from src.utils.main_utils import read_config_file
from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch


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
            
            logger.info(f"Warming up embedding model: {self.config['embedding']['model_name']}")
            _ = self.embedding_model.embed_query("warmup") 
            logger.info("Embedding model warm-up complete")

            
        except Exception as e:
            raise MyException(e,sys)        
    
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
                f"Vector store '{self.config['embedding']['db_name']}' "
                f"at {self.config['embedding']['db_path']} is empty (0 vectors)"
            )
            raise MyException("Vector store is empty", sys)
        
        logger.info(f"Vector store loaded with {count} vectors")
        
        return collection
    
    
    def retrieve_invoke(self,user_query):
        try: 
            return self.retriever.invoke(user_query)
        except Exception as e:
            raise MyException(e,sys)

        
    def retrive_adjacent_docs(self, user_query):
        """
        retrive pages from the vector store that are adjacent to the pages retrieved by the user query.
        """
        pass

class Reranker:
    
    def __init__(self,rerank_k:int):
        path = Path(r"D:\LegalSaathi AI\src\config\config_file.yaml")
        
        self.config = read_config_file(path)
        self.rerank_k = rerank_k
        self.reranker_model = self._load_reranker_model()


    def _load_reranker_model(self):
        return HuggingFaceCrossEncoder(
                model_name=self.config["rerank"]["model_name"],
                model_kwargs={
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
            )
    
    
    def rerank(self, docs: list, query: str) -> list:
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]

        predictions = self.reranker_model.client.predict(
            pairs,
            batch_size=16,
        )

        scored = sorted(
            zip(predictions, docs),
            key=lambda item: item[0],
            reverse=True,
        )

        return scored[:self.rerank_k]
