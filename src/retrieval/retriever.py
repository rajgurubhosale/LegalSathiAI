
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.main_utils import read_config_file
from langchain_chroma import Chroma
from src.logger import *
from src.exception import *
import sys

class Retrieval:
    def __init__(self):
        try:
            self.config = read_config_file()        
            self.top_k  = self.config['retrieval']['top_k']
            self.embed_model = self._load_model()
            self.vectorstore = self._load_vector_db()
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            
            logger.info(f"Warming up embedding model: {self.config['embedding']['model_name']}")
            _ = self.embed_model.embed_query("warmup") 
            logger.info("Embedding model warm-up complete")

            
        except Exception as e:
            raise MyException(e,sys)        
    
    def _load_model(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name = self.config['embedding']['model_name']
            )

        return embedding_model
    
    def _load_vector_db(self):
        collection = Chroma(
            collection_name=self.config['embedding']['db_name'],
            embedding_function=self.embed_model,
            persist_directory  = self.config['embedding']['db_path']   
        )
        return collection 
    
    
    def retrieve(self,user_query):
        try: 
            return self.retriever.invoke(user_query)
        except Exception as e:
            raise MyException(e,sys)

