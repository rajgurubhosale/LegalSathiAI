from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from langchain_core.documents import Document
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from string import ascii_lowercase
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from pathlib import Path
import re
from src.exception import *
from src.logger import *
import sys
import torch
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
login(token=os.getenv('HF_TOKEN'))

class PdfParser:
    def _build_document_converter(self):
        
        pipeline_options = PdfPipelineOptions()

        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False  
        accelerator = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU

        pipeline_options.accelerator_options = AcceleratorOptions(
            device = accelerator
    

        )

                
        document_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        return document_converter
    

    def _parse_pdf_to_docling(self,converter: DocumentConverter,file_path:Path):
        """
        Parses a PDF file into a Docling document using the provided DocumentConverter.

        """
        self.file_path = file_path
        docling_doc = converter.convert(self.file_path).document

        return docling_doc
    

    def _serialize_pages_to_documents(self,docling_doc):
        """
        Serializes each page of a Docling document into a list of LangChain Document objects.
        """
        serializer = MarkdownDocSerializer(doc=docling_doc)
        documents = []

        for page_no in sorted(docling_doc.pages):
            result = serializer.serialize(pages={page_no})
            if not result.text.strip():
                continue

            source_pages = sorted({
                prov.page_no
                for item in result.get_unique_doc_items()
                for prov in item.prov
            })

            documents.append(
                Document(
                    page_content=result.text,
                    metadata={
                        "page": page_no,                 # Page this export belongs to
                        "pages": source_pages or [page_no],  # All contributing PDF pages
                    },
                )
            )

        return documents

    def run(self, file_path: Path) -> List[Document]:
        """Parse a PDF file and return one Document per non-empty page."""
        
        document_converter = self._build_document_converter()

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        docling_doc = self._parse_pdf_to_docling(document_converter,file_path)

        return self._serialize_pages_to_documents(docling_doc)

class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator="start",
        )


    def assign_chunk_ids(self, split_docs,pdf_number:int):
        
        # Track how many chunks we've seen per page, to assign a/b/c...

        page_counters = {}

        for doc in split_docs:
            page = doc.metadata.get("page", doc.metadata.get("pages", ["unknown"]))
            # if page is a list (like [19, 20]), use the first page number as the base id
            page_key = page[0] if isinstance(page, list) else page

            count = page_counters.get(page_key, 0)
            suffix = ascii_lowercase[count] if count < 26 else f"z{count}"  # fallback if >26 chunks on one page

            # Use the filename for chunk identification, along with the page number and suffix
            doc.metadata["chunk_id"] = (
                f"{doc.metadata['act_name']}_{page_key}{suffix}"
            )
            page_counters[page_key] = count + 1

        return split_docs

    def run(self, documents: List[Document],pdf_number:int):
        """
        Full pipeline: split documents into chunks, then assign chunk IDs.
        """

        split_docs = self.splitter.split_documents(documents)

        chunked_docs = self.assign_chunk_ids(split_docs,pdf_number)

        return chunked_docs


class VectorStoreManager:
    """Handles creation and management of the Chroma vector store."""

    def __init__(self, collection_name="legal_saathi", persist_directory=r"D:\LegalSaathi AI\vectorstore", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.store = None
        self.embedding_model = HuggingFaceEmbeddings(
                            model_name=embedding_model_name,
                            encode_kwargs={"normalize_embeddings": True},
                        )
        
        self.store = self._get_or_create_collection()

    def _get_or_create_collection(self):

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )

        return self.store
    

    def get_embedding_model(self):
        return self.embedding_model

    def add_documents(self, documents):
        self.store.add_documents(
            documents,
            ids=[doc.metadata["chunk_id"] for doc in documents],
        )

    
    def run(self, chunked_docs: List[Document]):
        self.add_documents(chunked_docs)  
            

class LegalDocumentIngestionPipeline:

    def __init__(self, pdf_folder_path: Path, chunk_size: int = 800, chunk_overlap: int = 150):
        self.pdf_folder_path = Path(pdf_folder_path)

        self.parser= PdfParser()
        self.document_converter = self.parser._build_document_converter()

        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.vector_store_manager = VectorStoreManager()

        self.LABOUR_PDFS = {
                "The Code on Social Security, 2020.pdf",
                "The Code on Wages, 2019.pdf",
                "The Industrial Relations Code, 2020.pdf",
                "The Occupational Safety, Health and Working.pdf"
            }
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("<!-- image -->", "")
        text = text.replace("\u00a0", " ")  # Non-breaking space

        # Remove lines consisting only of underscores, including escaped \_
        text = re.sub(r"(?m)^[ \t]*(?:\\?_[ \t]*){3,}$", "", text)

        text = text.replace("IndiaCode", "")  # Remove standalone IndiaCode watermark

        # Reduce excessive blank lines
        text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

        return text.strip()
    
    def perform_ingestion(self):
        pdf_files = sorted(self.pdf_folder_path.glob("*.pdf"))
        total_pdfs = len(pdf_files)

        for pdf_number, pdf_file in enumerate(pdf_files, start=1):
            try:
                logger.info( f"Processing PDF {pdf_number}/{total_pdfs}: {pdf_file.name}")

                #Parse PDF
                docling_doc = self.parser._parse_pdf_to_docling(
                    self.document_converter, pdf_file
                )

                #Create page documents 
                documents = self.parser._serialize_pages_to_documents(docling_doc)

                #Add metadata BEFORE chunking
                category = ("labour" if pdf_file.name in self.LABOUR_PDFS else "other")

                for doc in documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    doc.metadata["category"] = category
                    doc.metadata["act_name"] = pdf_file.stem

                # Remove documents that became empty after cleaning
                documents = [doc for doc in documents if doc.page_content]  
                
                # 4. Split into chunks; metadata is copied into each chunk
                chunked_docs = self.chunker.run(documents, pdf_number)

                # 5. Store chunks
                self.vector_store_manager.run(chunked_docs)

                logger.info(
                    f"Completed PDF {pdf_number}/{total_pdfs}: {pdf_file.name}"
                )

            except Exception as e:
                logger.exception(f"Failed processing PDF: {pdf_file.name}")
                raise MyException(e, sys) from e


from pathlib import Path
from src.data_ingestion.ingest_and_chunk import LegalDocumentIngestionPipeline

if __name__ == "__main__":
    pdf_folder = Path(r"D:\LegalSaathi AI\PDF_DATA")  

    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder does not exist: {pdf_folder}")

    pipeline = LegalDocumentIngestionPipeline(
        pdf_folder_path=pdf_folder,
        chunk_size=800,
        chunk_overlap=150
    )

    pipeline.perform_ingestion()
    print("Ingestion completed successfully.")