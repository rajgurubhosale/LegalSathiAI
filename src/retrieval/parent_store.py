import json
from src.utils.main_utils import read_config_file
from src.logger import *


class ParentStore:
    def __init__(self):
        
        self.config = read_config_file()
        self.path = self.config['paths']['parent_store_path']
        
        # load the json file into memory when the object is created
        self.data = self._load_file()

    def _load_file(self):
        # open the json file and read it
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get(self, parent_id):
        # check if this parent_id exists in our json data
        if parent_id in self.data:
            return self.data[parent_id]["full_text"]
        else:
            logger.warning('The parent id is not preset look into it')
            return None

    def get_many(self, parent_ids):
        # this will store the final text list
        result = []
        # this will keep track of ids we already added (avoid duplicates)
        already_added = []

        for pid in parent_ids:
            if pid in already_added:
                continue  # skip if we already added this one

            text = self.get(pid)

            if text is not None:
                result.append(text)
                already_added.append(pid)

        return result