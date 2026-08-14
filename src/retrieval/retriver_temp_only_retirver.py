
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.utils.main_utils import read_config_file
from langchain_chroma import Chroma
from src.logger import *
from src.exception import *
import sys


class Retrieval:
    def __init__(self, top_n):
        try:
            self.config = read_config_file()
            self.top_n = top_n
            self.embed_model = self._load_model()
            self.vectorstore = self._load_vector_db()

            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_n}
            )

            logger.info(f"Warming up embedding model: {self.config['embedding']['model_name']}")
            _ = self.embed_model.embed_query("warmup")
            logger.info("Embedding model warm-up complete")

            # BM25 needs raw text, not vectors, so we build it from
            # the same docs already sitting in the Chroma collection.
            self.bm25_retriever = self._load_bm25_retriever()

        except Exception as e:
            raise MyException(e, sys)

    def _load_model(self):
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['embedding']['model_name']
        )

        return embedding_model

    def _load_vector_db(self):
        collection = Chroma(
            collection_name=self.config['embedding']['db_name'],
            embedding_function=self.embed_model,
            persist_directory=self.config['embedding']['db_path']
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

    def _load_bm25_retriever(self):
        # Pull everything back out of Chroma as plain LangChain Documents
        raw = self.vectorstore.get(include=["documents", "metadatas"])

        docs = []
        for text, meta in zip(raw["documents"], raw["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta or {}))

        if not docs:
            logger.error("No documents found to build BM25 index")
            raise MyException("BM25 corpus is empty", sys)

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.top_n

        logger.info(f"BM25 retriever built with {len(docs)} documents")

        return bm25

    def retrieve_invoke(self, user_query):
        try:
            return self.retriever.invoke(user_query)
        except Exception as e:
            raise MyException(e, sys)

    def bm25_retrieve_invoke(self, user_query):
        try:
            return self.bm25_retriever.invoke(user_query)
        except Exception as e:
            raise MyException(e, sys)

    def hybrid_retrieve_invoke(self, user_query, rrf_k: int = 60):
        """
        Combines dense (Chroma) + sparse (BM25) results using
        Reciprocal Rank Fusion (RRF).

        score(doc) = sum( 1 / (rrf_k + rank) ) across every retriever
        the doc appears in. Higher score = better. No need to
        normalize BM25 scores vs cosine scores this way.
        """
        try:
            dense_docs = self.retrieve_invoke(user_query)
            sparse_docs = self.bm25_retrieve_invoke(user_query)

            scores = {}   # key -> fused score
            doc_map = {}  # key -> actual Document object

            def add_ranks(docs):
                for rank, doc in enumerate(docs, start=1):
                    key = doc.page_content
                    scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank)
                    doc_map[key] = doc

            add_ranks(dense_docs)
            add_ranks(sparse_docs)

            # sort by fused score, highest first
            ranked_keys = sorted(scores, key=scores.get, reverse=True)

            fused_docs = [doc_map[key] for key in ranked_keys[:self.top_n]]

            logger.info(
                f"Hybrid retrieval: {len(dense_docs)} dense + "
                f"{len(sparse_docs)} sparse -> {len(fused_docs)} fused results"
            )

            return fused_docs

        except Exception as e:
            raise MyException(e, sys)