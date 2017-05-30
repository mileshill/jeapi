import json
import os
import shutil


class Reconstruct:
    """Reformat existing JSON structure into a hierarchical directory structure:
    data/
        nyiso/
            pe-{utility}.json
            ...
        pjm/
            pe-{utility}.json
    """
    def __init__(self, file_name: str):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = '/'.join([self.root_dir, file_name])
        self.data = self.load_file(self.file_path)

    @staticmethod
    def load_file(file_path: str) -> dict:
        """Loads the existing JSON structure
        :param file_path:
        :returns data
        """
        with open(file_path) as f:
            data = json.load(f)
        return data

    def load_iso_and_utility_names(self) -> dict:
        """returns ISOs and Utilities for directory and file creation
        """
        structure = {}
        for iso in self.data.keys():
            structure[iso.lower()] = [x.lower() for x in self.data[iso].keys()]
        return structure

    def check_or_create_directory(self) -> None:
        """Checks for directory structure; creates if !exists"""
        data_directory = self.root_dir + '/data'
        structure = self.load_iso_and_utility_names()
        iso_directories = [data_directory + '/' + iso for iso in structure.keys()]

        # remove existing ./data structure and child elements
        if os.path.exists(data_directory):
            shutil.rmtree(data_directory)

        # creates ./data
        os.makedirs(data_directory)

        # creates ./data/{iso}
        for iso_dir in iso_directories:
            os.makedirs(iso_dir)
        return

    def write_utilities_to_directory(self) -> None:
        """Deconstructs original JSON into utility specific files.
        Files are written into respective ISO directory.
        ./data/{iso}/pe-{utility}.json
        """
        for iso in self.data.keys():
            for utility in self.data[iso].keys():
                local_dir = '/'.join([self.root_dir, 'data', iso.lower(), 'pe-' + utility.lower()]) + '.json'
                file_data = {utility: self.data[iso][utility]}
                with open(local_dir, 'w') as f:
                    json.dump(file_data, f)

        return

    def execute_file_parse(self) -> None:
        """Primary method. Handle directory creation and JSON decomposition"""
        self.check_or_create_directory()
        self.write_utilities_to_directory()
        return

if __name__ == '__main__':
    r = Reconstruct(file_name='premise_explorer.json')
    r.execute_file_parse()