import pymupdf
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import re
import sys
from src.logger import *
import re
from src.utils.main_utils import *
import json
from src.exception import *
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice


class FileLoader:
    """Handles loading of PDF and markdown files."""

    def load_pdf_pymupdf(self, file_path):
        try:
            file = pymupdf.open(file_path)
            logger.info(f"Loaded PDF with pymupdf: {file_path}")
            return file
        except Exception as e:
            logger.error(f"Failed to load PDF with pymupdf: {file_path}")
            raise MyException(e, sys)

    def load_pdf_llm(self, file_path):
        try:
            loader = PyMuPDF4LLMLoader(file_path=file_path, mode="single")
            documents = loader.load()
            logger.info(f"Loaded PDF with LLM loader: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load PDF with LLM loader: {file_path}")
            raise MyException(e, sys)

    def load_markdown(self, file_path):
        try:
            with open(file_path, encoding="utf8") as f:
                content = f.read()
            logger.info(f"Loaded markdown file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Failed to load markdown file: {file_path}")
            raise MyException(e, sys)

        
    
        

class BNSCleaner():
    
    """Cleans BNS legal documents."""


    def __init__(self):        
        # file loader
        self.loader = FileLoader()


    
    def clean_first_section(self, input_path, output_path):
    
        documents = self.loader.load_pdf_llm(input_path)
        
        clean_text = ''
        
        page_split =documents[0].page_content.split('\n\n\n')

        temp_page = []
        for line in page_split:
            line = re.sub(r'\n|—|–|-----', ' ', line)
            line = re.sub(r'\*\*|––|;', '', line)
            temp_page.append(line)
        clean_text +=' '.join(temp_page)
            
        pattern = r'(\(\d+\))'
        pieces = re.split(pattern, clean_text)
        
        # split produces: ['intro', '(1)', 'text...', '(2)', 'text...']
        # pairs marker + content → ['(1) text...', '(2) text...']
        small_chunks = []
        for i in range(1, len(pieces), 2):
            small_chunks.append(pieces[i] + pieces[i+1])
            
            
        small_chunks_list = [txt for txt in small_chunks if len(txt) >= 20]

        # save the smal chunks list into json file
        
        ensure_path(output_path)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(small_chunks_list, f, ensure_ascii=False, indent=2)

    
    def load_and_merge_text(self, file_path):
        """Loads PDF text and fills missing pages using fallback."""
        

        
        documents = self.loader.load_pdf_llm(file_path)
        fallback_pdf = self.loader.load_pdf_pymupdf(file_path)

        for i, doc in enumerate(documents):
            char_count = len(doc.page_content.strip())
            if char_count < 20:
                print(f"Page index {i}: {char_count} characters -> using pymupdf fallback")
                doc.page_content = fallback_pdf[i].get_text().strip()

        fallback_pdf.close()

        # Merge all pages into one single string, in original order
        full_text = "\n".join(doc.page_content.strip() for doc in documents)

        return full_text
    
    
    def clean_text(self,full_text,output_path):
        """Convert raw text into structured markdown."""

        # convert the section bold to [section] for easy title breakdown dont collide
        converted_text = re.sub(r'\*\*(\d+)\.\*\*', r'[\1]', full_text)

        # PyMuPDF4LLM sometimes splits bold markers (**) across lines creating broken fragments.
        # These replacements merge them into a single space.
        converted_text = converted_text.replace('**\n\n**', ' ')
        converted_text = converted_text.replace('**\n**', ' ')
        converted_text = converted_text.replace('** **', ' ')
    
        # Converts bold markdown (**text**) to headings (## text) and saves to file.     
        text = re.sub(r'\*\*([^*]+)\*\*', r'## \1', converted_text)

        # save the file into the md file 
        ensure_path(output_path)

        with open(output_path,mode="w",encoding='utf8') as f:
            f.write(text)
            
        logger.info(f"Saved interim markdown: {output_path}")

            
    def fix_markdown_structure(self, input_path, output_path):
        
        """Final markdown cleanup."""
        
        text = self.loader.load_markdown(input_path)
        
        # Step 1: convert ## 14. or ## 14 . → [14]
        text = re.sub(r'##\s*(\d+)\s*\.', r'[\1]', text)

        # Step 2: fix ## stuck to previous text — inject newline
        text = re.sub(r'([^\n])##', r'\1\n\n##', text)

        # Step 3: ALL CAPS ## lines → # (chapter level)
        text = re.sub(r'^##\s+([A-Z][A-Z\s,\.]+)$', r'# \1', text, flags=re.MULTILINE)

        # Step 4: bare ALL CAPS lines with no ## → add #
        text = re.sub(r'^([A-Z][A-Z\s,\.]{3,})$', r'# \1', text, flags=re.MULTILINE)


        # ------------------------
        
        # make CHAPTER X, XI, XII etc → # (H1), case-insensitive, any number of leading #'s
        text = re.sub(r'^#{1,6}\s*(CHAPTER\s+[IVXLCDM\d]+.*)$', r'# \1', text, flags=re.MULTILINE | re.IGNORECASE)

        # make OF CONTEMPTS... style subheadings → ## (H2) — but only if they're ALL CAPS
        # and NOT already a section title
        text = re.sub(r'^#\s+([A-Z][A-Z\s,\.]+)$', r'## \1', text, flags=re.MULTILINE)

        # "## Punishment for theft" → "### Punishment for theft"
        # only targets mixed case (Capital + lowercase), ignores ALL CAPS chapter titles
        text = re.sub(r'^##\s+([A-Z][a-z][^\n]+)$', r'### \1', text, flags=re.MULTILINE)
                
        # "## 14. Whoever commits..." → "[14] Whoever commits..."
        # "### 14. Whoever commits..." → "[14] Whoever commits..."
        # works on any heading level (# to ######)
        text = re.sub(r'^#{1,6}\s*(\d+)\s*\.\s*(.+)$', r'[\1] \2', text, flags=re.MULTILINE)    


        # convert  ## CHAPTER -> # CHAPTER

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        pattern = re.compile(
            r'^#{1,6}[ \t]*(CHAPTER\s+[IVXLCDM\d]+.*)$',
            flags=re.MULTILINE | re.IGNORECASE
        )
        text, n = pattern.subn(r'# \1', text)
        
        ensure_path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.info(f"Saved final markdown: {output_path}")
    
    
class BNSSCleaner():
    
    def __init__(self):
        self.loader = FileLoader()
        
        
    def crop_pdf(self,input_path,output_path):
        
        pdf = self.loader.load_pdf_pymupdf(input_path)
        
        for page in pdf:
            rect = page.rect
            
            new_rect = pymupdf.Rect(
            rect.x0 + 110,  # Trim Left
            rect.y0 ,  # Trim Top
            rect.x1 - 110,  # Trim Right
            rect.y1    # Trim Bottom
            )
            
            page.set_cropbox(new_rect)
            
        ensure_path(output_path)

        pdf.save(output_path)
        pdf.close()

    def convert_to_markdown(self, input_path, output_path):
        '''
        convert text file into markdown
        # THE CODE IS RUNNED INTO KAGGLE
        
        '''
           
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False                  # skip OCR = much faster
        pipeline_options.do_table_structure = False        # detect tables
        
        pipeline_options.accelerator_options = AcceleratorOptions(

            num_threads=4,
            device=AcceleratorDevice.AUTO
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(input_path)

        md = result.document.export_to_markdown()
        ensure_path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
            
            
    def fix_markdown_structure(self, input_path, output_path):
        '''
        """Fix structure converted markdown."""
        
        '''

        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # strict: only matches "52." or "35." style — digits + dot, at start of line
        text = re.sub(r'(?m)^(\d+)\.\s', r'### \1\n\n', text)

        # converrt  ## CHAPTER -> # CHAPTER

        text = re.sub(r'^#{1,6}\s*(CHAPTER\s+[IVXLCDM\d]+.*)$', r'# \1', text, flags=re.MULTILINE | re.IGNORECASE)

        # create proper ## title for the sections
        text = re.sub(
            r'(# CHAPTER [IVXLCDM]+\n\n+)([A-Z][A-Z\s,]+)',
            r'\1## \2',
            text
        )
        
        ensure_path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        
    
    # SECOND SECTION
    def extract_tables_to_csv(self, input_path, output_csv_path):
        """Extract tables from PDF, clean them, and save as CSV."""
  
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True        # keep TRUE for tables!
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        result = converter.convert(input_path)

        # collect all tables into one dataframe
        all_dfs = []
        for table in result.document.tables:
            df = table.export_to_dataframe()
            all_dfs.append(df)

        interim_df = pd.concat(all_dfs, ignore_index=True)

        
        # clean first section of csv
        
        temp_df_1 = interim_df.loc[interim_df.iloc[:,:6].notna().all(axis=1)]
        clean_df_1 = temp_df_1.dropna(axis=1,how='all')
              
        # clean second section of csv  
        temp_df_1 = interim_df.loc[interim_df.iloc[:,12:-2].notna().all(axis=1)]
        clean_df_2 = temp_df_1.dropna(axis=1,how='all')
        
        # fix the columns name structure
        clean_df_2.columns = clean_df_1.columns
        
        # merge both cleaned df into final df
        final_df  = pd.concat([clean_df_1, clean_df_2], axis=0, ignore_index=True)
        ensure_path(output_csv_path)
        final_df.to_csv(output_csv_path,index=False)
        
        
        
class Pipeline:
    
    def __init__(self):
        logger.info("Initializing Pipeline")

        self.config = read_config_file()
        self.bns = BNSCleaner()
        self.bnss = BNSSCleaner()
        

    
    def _pdf_split(self, main_pdf_path, output_path, from_page, to_page):
        try:
            logger.info(f"Splitting PDF: {main_pdf_path} ({from_page}-{to_page})")
            main_src = pymupdf.open(main_pdf_path)
            new_file = pymupdf.open()

            new_file.insert_pdf(main_src, from_page=from_page, to_page=to_page)
            
            ensure_path(output_path)
            
            new_file.save(output_path)

            new_file.close()
            main_src.close()

            logger.info(f"Saved split file to {output_path}")

        except Exception as e:
            logger.error("Error during PDF split")
            raise MyException(e, sys)
        
        
    def _split_bns(self) -> None:
        src = self.config['paths']['pdf_paths']['bns']
        self._pdf_split(src, self.config['paths']['splits']['bns']['first_section'],  74,  81)
        self._pdf_split(src, self.config['paths']['splits']['bns']['second_section'], 82, 300)


    def _split_bnss(self) -> None:
        src = self.config['paths']['pdf_paths']['bnss']
        self._pdf_split(src, self.config['paths']['splits']['bnss']['first_section'],  17, 173)
        self._pdf_split(src, self.config['paths']['splits']['bnss']['second_section'], 174, 203)

        
    def _clean_bns(self) -> None:
        # first section → JSON chunks
        self.bns.clean_first_section(
            input_path = self.config['paths']['splits']['bns']['first_section'],
            output_path             = self.config['paths']['final']['bns']['first_section']
        )

        # second section → interim MD → final MD
        second_section_path = self.config['paths']['splits']['bns']['second_section']
        interim_path        = self.config['paths']['interim']['bns']['main']

        full_text = self.bns.load_and_merge_text(second_section_path)
        self.bns.clean_text(full_text, interim_path)
        self.bns.fix_markdown_structure(interim_path, self.config['paths']['final']['bns']['second_section'])

    def _clean_bnss(self) -> None:
        splits   = self.config['paths']['splits']['bnss']
        interim  = self.config['paths']['interim']['bnss']
        final    = self.config['paths']['final']['bnss']

        # first section → crop → convert → fix structure
        self.bnss.crop_pdf(splits['first_section'], interim['sections_cropped'])
        self.bnss.convert_to_markdown(interim['sections_cropped'], interim['sections'])
        self.bnss.fix_markdown_structure(interim['sections'], final['sections'])

        # second section → extract tables
        self.bnss.extract_tables_to_csv(splits['second_section'], final['tables'])

    def run(self) -> None:
        try:
            logger.info("Pipeline started")

            self._split_bns()
            self._split_bnss()

            self._clean_bns()
            self._clean_bnss()

            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.critical("Pipeline failed")
            raise MyException(e, sys)


if __name__ == '__main__':
    runner = Pipeline()
    runner.run()