from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from langchain_core.documents import Document
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from string import ascii_lowercase
from typing import List
from pathlib import Path
import re
from src.exception import *
from src.logger import *    
import sys
import torch
from dotenv import load_dotenv
from src.utils.main_utils import read_config_file
import json
load_dotenv()

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
        docling_doc = converter.convert(file_path).document

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
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 100):
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

def clean_text(text: str) -> str:
    text = text.replace("<!-- image -->", "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(?m)^[ \t]*(?:\\?_[ \t]*){3,}$", "", text)
    text = text.replace("IndiaCode", "")
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def main():
    cfg = read_config_file()["ingestion"]

    pdf_folder = Path(cfg["pdf_folder"])
    chunks_path = Path(cfg["chunks_path"])
    labour_pdfs = set(cfg["labour_pdfs"])
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_folder}")

    parser = PdfParser()
    converter = parser._build_document_converter()      # built once, reused
    chunker = Chunker(cfg["chunk_size"], cfg["chunk_overlap"])

    total, failed = 0, []

    with open(chunks_path, "w", encoding="utf-8") as f:   # "w": no duplicates on re-run
        for number, pdf in enumerate(pdf_files, start=1):
            try:
                logger.info(f"[{number}/{len(pdf_files)}] {pdf.name}")

                docling_doc = parser._parse_pdf_to_docling(converter, pdf)
                pages = parser._serialize_pages_to_documents(docling_doc)

                for doc in pages:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata["act_name"] = pdf.stem
                    doc.metadata["category"] = "labour" if pdf.name in labour_pdfs else "other"
                pages = [d for d in pages if d.page_content]

                chunks = chunker.run(pages, number)

                for c in chunks:
                    f.write(json.dumps(
                        {"page_content": c.page_content, "metadata": c.metadata},
                        ensure_ascii=False) + "\n")

                total += len(chunks)
                logger.info(f"{pdf.name}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed on {pdf.name}: {e}")
                failed.append(pdf.name)

    logger.info(f"Saved {total} chunks to {chunks_path}")
    if failed:
        logger.warning(f"Failed files: {failed}")


if __name__ == "__main__":
    main()

    