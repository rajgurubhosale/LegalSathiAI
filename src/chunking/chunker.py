import re
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.utils.main_utils import read_config_file, ensure_path
from src.logger import logger
from src.exception import *
import json
import math
import tiktoken
import numpy as np
import pandas as pd



class BNSChunker:
    """Chunks cleaned BNS markdown into parent-child structure."""

    HEADERS = [
        ("#",   "chapter"),
        ("##",  "chapter_title"),
        ("###", "section_title"),
    ]


    def _clean_markdown(self, md_text):
        md_text = md_text.replace("### Illustrations._", "_Illustrations_.")
        return md_text
    
    
    def _split_markdown(self, md_text: str) -> list:
        """Split markdown by headers into sections."""
        
        md_text = self._clean_markdown(md_text)
        
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS,
            strip_headers=True
        )
        return splitter.split_text(md_text)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


    def _clean_text(self, text: str) -> str:
        """Remove noise characters from chunk text."""
        text = text.replace('\n\n', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('.–', ' ')
        text = text.replace('.—', ' ')
        text = re.sub(r' +', ' ', text)
        # remove ----- pattern
        text = re.sub(r'-{2,}', ' ', text)
        return text.strip()

    def _make_child_chunks(self, text):
        if self._count_tokens(text) <= 450:
            return [text]

        pieces = re.split(r'(?=_Illustrations_|_Illustration_)', text)  # removed _Explanation_
        if len(pieces) > 1:
            return pieces
        else:
            return [text]
            

    def _build_second_section_chunks(self, docs: list) -> dict:
        """Build parent abd children chunks, keyed by section number."""

        
        parent_data = {}
        child_data = {}

        for i, doc in enumerate(docs):
            text = self._clean_text(doc.page_content)

            if not text or len(text) < 20:
                continue

            match = re.search(r'^\[(\d+)\]', text)
            section_num = match.group(1) if match else str(i)
            
         
            
            parent_id = f'BNS_{section_num}'
               
            doc.metadata['act'] = 'BNS'
            doc.metadata['section'] = section_num

            doc.metadata['parent_id'] = parent_id

            parent_data[parent_id] = {
                "full_text": text
            }
        

            children = self._make_child_chunks(text)

            child_list = []
            
            for child_text in children:
                child_text = child_text.strip()
                if child_text:
                    child_list.append(child_text)
                
            child_data[parent_id] = {
                "children": child_list,
                "metadata":   doc.metadata
            }
                
        final_data = {
            'parent_data':parent_data,
            'children_data':child_data
        }

        return final_data

        

    def chunk(self, input_path: str, output_path: str) -> None:
        """Full chunking pipeline — read MD, chunk, save JSON."""
        logger.info(f"Chunking BNS file: {input_path}")

        with open(input_path, encoding="utf-8") as f:
            md_text = f.read()

        docs        = self._split_markdown(md_text)
        final_data = self._build_second_section_chunks(docs)
        
        ensure_path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved BNS second section chunks to {output_path}")
            
    def chunks_first_section(self, input_path: str, output_path: str) -> None:
        """Load first section JSON (raw list, e.g. definitions), convert into
        parent_data/child_data structure, save to output path."""

        logger.info(f"Processing BNS first_section: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        parent_data = {}
        child_data = {}

        for i, text in enumerate(data):
            key = f'BNS_DEF_{i}'

            parent_data[key] = {
                'full_text': text
            }

            child_data[key] = {
                'children': [text],
                'metadata': {
                    'chapter_title': 'PRELIMINARY',
                    'parent_id': key
                }
            }

        final_data = {
            'parent_data': parent_data,
            'children_data': child_data
        }

        ensure_path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved BNS first section chunks to {output_path}")
                
                
class BNSSChunker:
    """Chunks cleaned BNSS markdown into parent-child structure."""
    

    HEADERS = [
        ("#",   "chapter"),
        ("##",  "chapter_title"),
        ("###", "section_data"),
    ]
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def _decide_chunk_count(self, text: str) -> int:
        """Decide how many child chunks to split a section into based on token count."""
        token_count = self._count_tokens(text)
        
        if token_count <= 450:
            return 1
        
        for i in range(2, 10):
            if token_count / i < 450:
                return i
        return 9
    def _fix_broken_headers(self,md_text):
        """
        Some real headers get split across two lines by the PDF converter, e.g.:
            ## RECIPROCAL ARRANGEMENTS ... PROCEDURE FOR

            ATTACHMENT AND FORFEITURE OF PROPERTY
        This merges them back into a single '##' line.

        Also demotes fake headers like '## Illustrations to sub-section ( 3 )'
        by stripping the '##' so they stay as plain text instead of splitting
        the section apart.
        """
        pattern = r'^(##\s+[A-Z0-9 ,.\-]+)\n\n([A-Z0-9 ,.\-]+)$'
        md_text = re.sub(pattern, r'\1 \2', md_text, flags=re.MULTILINE)
        
        pattern = r'^##\s+(Illustrations?\s+to\s+sub-section.*)$'
        md_text = re.sub(pattern, r'\1', md_text, flags=re.MULTILINE)
        return md_text

    def _clean_child_text(self, text: str) -> str:
        """Remove noise from child chunk text."""
        text = text.replace('  ', ' ')
        text = text.replace('-   ', ' ')
        text = text.replace('-  ', ' ')
        
        text = re.sub(r'\r\n|\r|\n', ' ', text)
        text = text.replace('\n',' ')
        # replace actual double quotes with single quote
        
        text = text.replace('"', "'")
        
        # collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # remove bullet dash "- " → space (only dash followed by space, not hyphens in words)
        text = re.sub(r'(?<!\w)-\s', ' ', text)
        
        return text.strip()

    def _clean_parent_text(self, text: str) -> str:
        """Remove noise from parent text."""
        text = text.replace('\n', ' ')
        text = text.replace('_', '')
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def _split_markdown(self, md_text: str) -> list:
        """Split markdown by headers into sections."""
        md_text= self._fix_broken_headers(md_text)
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS,
            strip_headers=True
        )
        return splitter.split_text(md_text)
    
    def _make_child_chunks(self, doc: str) -> list[str]:
        """Split section text into child chunks based on token size.
        doc: single doc page from docs
        
        """
        num_splits    = self._decide_chunk_count(doc)
        sentence_chunks = doc.split('\n')

        if num_splits > 1:
            chunk_size = math.floor(len(sentence_chunks) / num_splits)
            
            if chunk_size == 0:
                return [self._clean_child_text(doc)]
            
            raw_chunks = [
                sentence_chunks[i: i + chunk_size]
                for i in range(0, len(sentence_chunks), chunk_size)
            ]
            
            children = []
            
            for chunk in raw_chunks:
                joined  = ' '.join(chunk)
                cleaned = self._clean_child_text(joined)
                children.append(cleaned)
            return children
        else:
            return [self._clean_child_text(doc)]  
        
    def _build_sections_chunks(self, docs: list) -> dict:
        """Build parent and children chunks for BNSS, keyed by parent_id."""

        parent_data = {}
        child_data = {}

        i = 0
        for doc in docs:
            i+=1
            text = doc.page_content.strip()

            if not text or len(text) < 20:
                continue

            # try [52] marker first
            match = re.search(r'^\[(\d+)\]', text)


            if match:
                section_num = match.group(1)
            else:
                section_num = doc.metadata.get("section_data") or doc.metadata.get("chapter_title")
                if not section_num:
                    section_num = f"unknown_{i}"

            
            parent_id = f'BNSS_{section_num}'

            doc.metadata['act'] = 'BNSS'
            doc.metadata['section'] = section_num
            doc.metadata['parent_id'] = parent_id

            parent_text = self._clean_parent_text(text)

            parent_data[parent_id] = {
                "full_text": parent_text
            }

            children = self._make_child_chunks(text)

            child_list = []
            for child_text in children:
                child_text = child_text.strip()
                if child_text:
                    child_list.append(child_text)

            child_data[parent_id] = {
                "children": child_list,
                "metadata": doc.metadata
            }

        final_data = {
            'parent_data': parent_data,
            'children_data': child_data
        }

        return final_data

    
    def chunk_first_section(self, input_path: str, output_path: str) -> None:
        """Full chunking pipeline — read MD, chunk, save JSON."""
        logger.info(f"Chunking BNSS file: {input_path}")

        with open(input_path, encoding="utf-8") as f:
            md_text = f.read()

        docs        = self._split_markdown(md_text)
        final_data = self._build_sections_chunks(docs)
        
        ensure_path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved BNSS sections chunks to {output_path}")
        
    
    def _clean_common(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip <br> tags and normalize Ditto markers to NaN (shared by both tables)."""
        df = df.replace(r'<br\s*/?>', ' ', regex=True)
        df = df.replace({'Ditto': np.nan, 'Ditto.': np.nan})
        return df

   
    
    def _load_other_laws(self, input_path: str) -> pd.DataFrame:
        """Load + clean the 'offences against other laws' CSV."""
        df = pd.read_csv(input_path)
        if 'Section' in df.columns:
            df = df.drop(columns=['Section'])
        df = self._clean_common(df)
        df = df.ffill()
        return df
    
    def _extract_clean_section(self, section: str) -> str:
        """Extract the main section number from a Section value like '356(2)' -> '356'."""
        match = re.match(r'^(\d+)', str(section).strip())
        return match.group(1) if match else str(section).strip()

    def _load_main_table(self, input_path: str) -> pd.DataFrame:
        """Load + clean the main BNSS schedule table CSV."""
        df = pd.read_csv(input_path)
        df = self._clean_common(df)

        df['Section'] = df['Section'].ffill()
        df['Offence'] = df['Offence'].ffill()
        df = df.ffill()

        df['clean_section'] = df['Section'].apply(self._extract_clean_section)
        return df

    def _build_row_line(self, row: pd.Series) -> str:
        """One line of text for a single row (used for both single & sub-section cases)."""
        return (
            f"{row['Section']}: {row['Offence']}. "
            f"Punishment: {row['Punishment']}. "
            f"Cognizable: {row['Cognizable']}. "
            f"Bailable: {row['Bailable']}. "
            f"Court: {row['Court']}"
        )

    def _build_chunk_for_group(self, group: pd.DataFrame) -> str:
        """Build chunk text for one clean_section group."""
        if len(group) == 1:
            row = group.iloc[0]
            return self._build_row_line(row)
        else:
            main_section = group.name
            header = f"Main Section - {main_section}"
            lines = [self._build_row_line(row) for _, row in group.iterrows()]
            return header + "\n" + "\n".join(lines)

    def _build_main_table_chunks(self, df: pd.DataFrame) -> tuple:
        """Build parent_data/children_data for the main table, one entry per clean_section."""
        parent_data = {}
        children_data = {}

        chunks = (
            df.groupby('clean_section', sort=False)
            .apply(self._build_chunk_for_group)
            .reset_index(name='chunk_text')
        )

        for _, row in chunks.iterrows():
            section = row['clean_section']
            text = row['chunk_text']
            parent_id = f"BNSS_schedule_1_{section}"

            metadata = {
                "act": "BNS",
                "source_act": "BNSS",
                "schedule_source": "BNSS Schedule I",
                "section": str(section),
                "type": "schedule_1",
                "parent_id": parent_id
            }

            parent_data[parent_id] = {"full_text": text}
            children_data[parent_id] = {"children": [text], "metadata": metadata}

        return parent_data, children_data

    def _row_to_chunk_general(self, row: pd.Series) -> str:
        """Convert an 'other laws' row (no Section number) into chunk text."""
        return (
            f"Offence: {row['Offence']}. "
            f"Cognizable: {row['Cognizable']}. "
            f"Bailable: {row['Bailable']}. "
            f"Triable by: {row['Court']}"
        )
        
        
    
    # ---------- building parent/child chunks ----------

    def _build_other_laws_chunk(self, other_df: pd.DataFrame) -> tuple:
        """Build a single combined parent/child chunk for the 'other laws' block."""
        header = "I. CLASSIFICATION OF OFFENCES AGAINST OTHER LAWS"

        other_df = other_df.copy()
        other_df['chunk_text'] = other_df.apply(self._row_to_chunk_general, axis=1)
        
        rows_text = " | ".join(other_df['chunk_text'].tolist())
        other_text = f"{header}. {rows_text}"
        other_parent_id = "BNSS_schedule_1_end"

        other_metadata = {
            "act": "BNSS",
            "section": "other_laws",
            "type": "schedule_1",
            "parent_id": other_parent_id
        }

        parent_entry = {other_parent_id: {"full_text": other_text}}
        child_entry = {
            other_parent_id: {"children": [other_text], "metadata": other_metadata}
        }
        return parent_entry, child_entry

    # ---------- pipeline entrypoint ----------

    def chunk_second_section(
        self, main_table_path: str, other_laws_path: str, output_path: str
    ) -> None:
        logger.info(f"Chunking BNSS main table: {main_table_path}")
        main_df = self._load_main_table(main_table_path)
        parent_data, children_data = self._build_main_table_chunks(main_df)   # <-- pass main_df directly, no merge step

        logger.info(f"Chunking BNSS other-laws table: {other_laws_path}")
        other_df = self._load_other_laws(other_laws_path)
        other_parent, other_child = self._build_other_laws_chunk(other_df)
        parent_data.update(other_parent)
        children_data.update(other_child)

        final_data = {"parent_data": parent_data, "children_data": children_data}

        ensure_path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(parent_data)} table chunks to {output_path}")

class Pipeline:
    
    def __init__(self):
        logger.info('Chunking pipelien initialized')
        self.config  = read_config_file()

        self.bns_chunker = BNSChunker()  
        self.bnss_chunker = BNSSChunker()

    def _chunk_bns(self) -> None:
        self.bns_chunker.chunk(
            input_path  = self.config['paths']['final']['bns']['second_section'],
            output_path = self.config['paths']['chunks']['bns']['second_section']
        )
        
        self.bns_chunker.chunks_first_section(
            input_path  = self.config['paths']['final']['bns']['first_section'],
            output_path = self.config['paths']['chunks']['bns']['first_section']
        )
    def _chunk_bnss(self) -> None:
        # sections → parent child chunks
        self.bnss_chunker.chunk_first_section(
            input_path  = self.config['paths']['final']['bnss']['sections'],
            output_path = self.config['paths']['chunks']['bnss']['sections']
        )
        
        self.bnss_chunker.chunk_second_section(
            main_table_path = self.config['paths']['final']['bnss']['tables'],
            other_laws_path = self.config['paths']['final']['bnss']['other_laws'],
            output_path      = self.config['paths']['chunks']['bnss']['tables']
        
        )   
        

    def run(self) -> None:
        
        try:
            logger.info('Chunking Started')
            self._chunk_bns()     
            logger.info("BNS chunking completed")
            self._chunk_bnss()
            logger.info("BNSS chunking completed")
            
            logger.info("Pipeline finished successfully")        
        except Exception as e:
            logger.critical('Chunking Pipeline Failed')
            raise MyException(e,sys)

if __name__ == '__main__':
    runner = Pipeline()
    runner.run()