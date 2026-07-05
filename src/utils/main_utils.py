import yaml
from src.exception import *
from src.logger import *
from pathlib import Path


def read_config_file():
    '''read and loads the yaml file from given path
    
    return:
        config.yaml: yaml_file from config
    '''
    try:
        
        yaml_file_path  = 'D:/LegalSaathi AI/src/config/config.yaml'
        
        with open(yaml_file_path,'r') as f:    
            file = yaml.safe_load(f)
            
        logger.info(f'Yaml file load from {yaml_file_path} succesfully')
        
        return file
    
    except FileNotFoundError as e:
        raise MyException(e,sys)
    

def ensure_path(path: str):
    """
    Ensures the parent directory of a file path exists.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)