from pathlib import Path
from src.utils.main_utils import read_config_file,ensure_path
import json
from src.logger import *
from src.exception import MyException
import sys


class ParentStore:
    """Loads and merges all parent_data sources into one in-memory lookup,
    used at generation time to swap child chunks for full parent context."""

    def __init__(self):
        self.config = read_config_file()

    def _get_paths(self) -> list:
        """Pulls all 4 chunk file paths from config."""
        return [
            self.config['paths']['chunks']['bns']['first_section'],
            self.config['paths']['chunks']['bns']['second_section'],
            self.config['paths']['chunks']['bnss']['sections'],
            self.config['paths']['chunks']['bnss']['tables'],
        ]
        
    def _load_and_merge(self, paths: list,output_path:str) -> dict:
        try:
            merged = {}
            for path in paths:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                merged.update(data['parent_data'])

            ensure_path(output_path)        
            with open(output_path,'w',encoding='utf8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            logger.info(f'final Merged parent data saved in {output_path}')
        except Exception as e:
            raise MyException(e,sys)        
        

    def run(self):
        """Loads and merges all parent data. Call once at startup."""
        
        paths = self._get_paths()
        output_path = self.config['parent_store_path']
        
        self._load_and_merge(paths,output_path)

        return self


if __name__ == '__main__':
    store = ParentStore()
    store.run()
    