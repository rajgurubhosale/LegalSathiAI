"""Small experiment: section-aware chunks with exact PDF-page provenance.

Run this file to ingest the two sample Acts into an isolated Chroma collection.
The full parent sections are saved as JSONL for expansion after retrieval.
"""

import json
import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "PDF_DATA"
STORE_DIR = ROOT / "vectorstore_section_test"
PARENT_FILE = STORE_DIR / "parent_sections.jsonl"
COLLECTION = "legal_saathi_section_test"
SAMPLE_PDFS = [
    "THE CHILD AND ADOLESCENT (PROHIBITION AND REGULATION) ACT, 1986.pdf",
    "The Code on Wages, 2019.pdf",
]

# The optional number and '[' cover amendment markers such as "1 [3. ...".
SECTION_START = re.compile(
    r"(?<!\w)(?:\d+\s*\[\s*)?(\d{1,3})([A-Z]?)\s*\.\s*"
    r"(?=[A-Z0-9][^\n]{0,250}?\.\s*[-–—])"
)


def extract_sections(pdf_path, converter):
    """Keep ordered text blocks and their physical PDF pages together."""
    doc = converter.convert(str(pdf_path)).document
    sections = []
    current = None
    current_number = (0, "")
    in_act = False

    blocks = []
    for item, _ in doc.iterate_items():
        if str(item.label) in {"footnote", "page_header", "page_footer"}:
            continue
        content = getattr(item, "text", "")
        if not content.strip() or not item.prov:
            continue
        if len({p.page_no for p in item.prov}) == 1:
            p = item.prov[0]
            blocks.append((p.page_no, -p.bbox.t, content.strip()))
        else:
            previous = 0
            for p in sorted(item.prov, key=lambda p: p.charspan[0]):
                left, right = p.charspan
                if content[previous:left].strip():
                    raise ValueError(f"Unmapped page text in {pdf_path.name}")
                blocks.append((p.page_no, -p.bbox.t, content[left:right].strip()))
                previous = right
            if content[previous:].strip():
                raise ValueError(f"Unmapped page text in {pdf_path.name}")

    # These sample Acts use one column. Sorting by page position fixes occasional
    # Docling reading-order swaps between adjacent provisions.
    for page, _, content in sorted(blocks):
        if not content:
            continue
        pages = [page]
        if not in_act:
            in_act = "ACT NO" in content.upper()
            continue
        if content.upper().lstrip("0123456789 [").startswith(
            ("THE SCHEDULE", "SCHEDULE", "STATEMENT OF OBJECTS AND REASONS")
        ):
            break

        start = 0
        for match in SECTION_START.finditer(content):
            number = (int(match.group(1)), match.group(2))
            if number <= current_number:
                continue
            if current and match.start() > start:
                current["parts"].append((content[start:match.start()].strip(), pages))
            if current:
                sections.append(current)
            current_number = number
            current = {
                "act_name": pdf_path.stem,
                "pdf": pdf_path.name,
                "section_id": f"{match.group(1)}{match.group(2)}",
                "parts": [],
            }
            start = match.start()
        if current and content[start:].strip():
            current["parts"].append((content[start:].strip(), pages))

    if current:
        sections.append(current)
    if not sections:
        raise ValueError(f"No numbered sections found in {pdf_path.name}")
    return sections


def make_chunks(section, splitter):
    """Split a section and assign only the pages touching each child chunk."""
    text_parts = []
    spans = []
    offset = 0
    for content, pages in section["parts"]:
        if text_parts:
            text_parts.append("\n\n")
            offset += 2
        start = offset
        text_parts.append(content)
        offset += len(content)
        spans.append((start, offset, pages))

    parent_text = "".join(text_parts)
    parent_pages = sorted({page for _, _, pages in spans for page in pages})
    parent = {
        "act_name": section["act_name"],
        "pdf": section["pdf"],
        "section_id": section["section_id"],
        "pages": parent_pages,
        "text": parent_text,
    }

    children = []
    for index, child in enumerate(splitter.create_documents([parent_text]), start=1):
        start = child.metadata["start_index"]
        end = start + len(child.page_content)
        pages = sorted({
            page
            for left, right, source_pages in spans
            if left < end and right > start
            for page in source_pages
        })
        if not pages:
            raise ValueError(f"Lost page provenance for {section['pdf']} section {section['section_id']}")
        child.metadata = {
            "act_name": section["act_name"],
            "pdf": section["pdf"],
            "section_id": section["section_id"],
            "pages": pages,
            "chunk_id": f"{section['act_name']}:{section['section_id']}:{index}",
        }
        children.append(child)
    return parent, children


def get_parent(act_name, section_id):
    """Expand a retrieved child using its act_name and section_id metadata."""
    with PARENT_FILE.open(encoding="utf-8") as file:
        for line in file:
            parent = json.loads(line)
            if parent["act_name"] == act_name and parent["section_id"] == section_id:
                return parent
    raise KeyError((act_name, section_id))


def main():
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    parents = []
    children = []
    for name in SAMPLE_PDFS:
        sections = extract_sections(PDF_DIR / name, converter)
        for section in sections:
            parent, section_children = make_chunks(section, splitter)
            parents.append(parent)
            children.extend(section_children)
        print(f"{name}: {len(sections)} sections")

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with PARENT_FILE.open("w", encoding="utf-8") as file:
        for parent in parents:
            file.write(json.dumps(parent, ensure_ascii=False) + "\n")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(STORE_DIR),
        embedding_function=embeddings,
    )
    # Rebuild only this experimental collection, so reruns leave no old chunks.
    store.reset_collection()
    # Chroma metadata cannot hold lists, so encode child pages as JSON.
    stored_children = [
        Document(page_content=child.page_content, metadata={
            **child.metadata,
            "pages": json.dumps(child.metadata["pages"]),
        })
        for child in children
    ]
    store.add_documents(stored_children, ids=[d.metadata["chunk_id"] for d in children])
    print(f"Stored {len(children)} child chunks in {COLLECTION}")
    print(f"Saved {len(parents)} parent sections in {PARENT_FILE}")


if __name__ == "__main__":
    main()
