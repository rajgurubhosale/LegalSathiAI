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

    def _split_markdown(self, md_text: str) -> list:
        """Split markdown by headers into sections."""
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

    def _make_child_chunks(self,text):
        """
        - if section is <= MAX_PARENT_TOKENS tokens: don't split at all
        - if it's over toknes and contains an _Explanation_/_Illustration_
        split it into child sections

        """
        if self._count_tokens(text) <= 450:
            return [text]

        pieces =  re.split(r'(?=_Explanation_|_Illustrations_|_Illustration_)', text)
        if len(pieces)> 1:
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
        return 1
    
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

        for doc in docs:
            text = doc.page_content.strip()

            if not text or len(text) < 20:
                continue

            # try [52] marker first
            match = re.search(r'^\[(\d+)\]', text)

            if match:
                section_num = match.group(1)
            else:
                # fallback to metadata section_data if available
                section_num = doc.metadata.get("section_data") or \
                            doc.metadata.get("chapter_title") or \
                            "unknown"

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
        
    def _load_and_clean_csv(self, input_path: str) -> pd.DataFrame:
        """Load CSV, drop header row, replace Ditto values."""
        df = pd.read_csv(input_path)
        df = df.drop(index=0).reset_index(drop=True)
        df.replace({'Ditto': np.nan, 'Ditto.': np.nan}, inplace=True)
        df.ffill(inplace=True)
        return df
    
    def _row_to_chunk(self, row: pd.Series) -> str:
        """Convert single table row to readable chunk text."""
        return (
            f"Section {row['Section']}: "
            f"{row['Offence']}. "
            f"Punishment: {row['Punishment']}. "
            f"Cognizable: {row['Cognizable or non- cognizable']}. "
            f"Bailable: {row['Bailable or Non- bailable']}. "
            f"Triable by: {row['By what Court triable']}"
        )

    def _build_table_chunks(self, df: pd.DataFrame) -> dict:
        """Build parent + children chunks from schedule table (main rows + other-laws block),
        same structure as section chunks: parent_data / children_data keyed by parent_id."""

        parent_data = {}
        child_data = {}

        # --- main table rows (0:343) ---
        df_clean = df.iloc[:343].copy().reset_index(drop=True)
        df_clean['chunk_text'] = df_clean.apply(self._row_to_chunk, axis=1)

        for _, row in df_clean.iterrows():
            parent_id = f"BNSS_schedule_1_{row['Section']}"
            text = row['chunk_text']

            metadata = {
                "act": "BNSS",
                "source_act": "BNS",
                "section": str(row['Section']),
                "type": "schedule_1",
                "parent_id": parent_id
            }

            parent_data[parent_id] = {
                "full_text": text
            }

            child_data[parent_id] = {
                "children": [text],          # single self-contained row, always a list
                "metadata": metadata
            }

        # --- "other laws" block (345:347) ---
        header = "I. CLASSIFICATION OF OFFENCES AGAINST OTHER LAWS"
        df_other = df.iloc[345:347].copy().reset_index(drop=True)

        rows_text = " | ".join([
            f"{row.iloc[2]}: "
            f"Cognizable: {row.iloc[3]}. "
            f"Bailable: {row.iloc[4]}. "
            f"Court: {row.iloc[5]}"
            for _, row in df_other.iterrows()
        ])

        other_text = f"{header}. {rows_text}"
        other_parent_id = 'BNSSschedule_1_end'

        other_metadata = {
            "act": "BNSS",
            "section": "other_laws",
            "type": "schedule_1",
            "parent_id": other_parent_id
        }

        parent_data[other_parent_id] = {
            "full_text": other_text
        }

        child_data[other_parent_id] = {
            "children": [other_text],
            "metadata": other_metadata
        }

        return {
            'parent_data': parent_data,
            'children_data': child_data
        }

    def chunk_second_section(self, input_path: str, output_path: str) -> None:
        """Full chunking pipeline — load CSV, chunk, save JSON."""
        logger.info(f"Chunking BNSS table: {input_path}")

        df = self._load_and_clean_csv(input_path)
        final_data = self._build_table_chunks(df)

        ensure_path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        parent_count = len(final_data['parent_data'])
        logger.info(f"Saved {parent_count} table chunks to {output_path}")
        
        
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
            input_path  = self.config['paths']['final']['bnss']['tables'],
            output_path = self.config['paths']['chunks']['bnss']['tables']
            
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