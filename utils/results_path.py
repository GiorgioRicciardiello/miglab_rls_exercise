
import pathlib
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
from typing import Union, Optional, Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import config_paths
from typing import Dict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

class ResultsPath:
    def __init__(self,result_main_dir:pathlib.Path,
                 response_split: str = 'ideal'):
        """
        The types of models to test are currently defined by the way we encode the target variable. Therefore, we create
        a class that will create the directory for the specific model we are testing
        :param result_main_dir: main directory of the result folder
        :param response_split:

        response_split = 'ideal'
        result_main_dir = config_paths.get('results_path')
        """
        self.result_main_dir = result_main_dir
        self.response_split = response_split
        self.result_path = None

    def create_results_dict(self):
        # we will always start with a file _1
        result_path = self.result_main_dir.joinpath(self.response_split + '_1')
        if result_path.exists():
            # the folder exists, so we will create a new one with the next possible suffix
            matching_folders, last_file_number = self._find_folders_with_suffix(
                base_path=self.result_main_dir,
                common_prefix=self.response_split,
            )

            if isinstance(last_file_number, int) and len(matching_folders) >= 1:
                # we have one already two ideas
                new_folder_number = last_file_number + 1
                self.result_path = self.result_main_dir.joinpath(self.response_split + f'_{new_folder_number}')

                # Check if self.result_path is already in matching_folders (Sanity check)
                if self.result_path in matching_folders:
                    raise ValueError(f"The result path {self.result_path} already exists in matching_folders."
                                     f"There is a problem with the suffix creation '_{new_folder_number}'")

                self.result_path.mkdir(parents=True, exist_ok=False)
                print(f'Making directory in \n {self.result_path}')
        else:
            # make the first file with the _1 identifier
            self.result_path = self.result_main_dir.joinpath(self.response_split + '_1')
            self.result_path.mkdir(parents=True, exist_ok=False)
            print(f'Making directory in \n {self.result_path}')

    @staticmethod
    def _find_folders_with_suffix(base_path:pathlib.Path,
                                  common_prefix:str) -> tuple[list[Path], int]:
        """
        Search all the folder in the base path that the folder with common prefix
        :param base_path: pathlib.Path, base path to search the directory
        :param common_prefix: str, suffix of the subfolder in the base path we are searching
        :return:
            matching_folders, list of paths
            last_file_number: int, number of the last folder with the same directory
        """
        # List all directories in the parent path
        all_directories = [entry for entry in base_path.iterdir() if entry.is_dir()]

        # Filter directories with the specified common prefix
        matching_folders = [folder for folder in all_directories if folder.name.startswith(common_prefix)]

        last_file_number = [int(entry.name.split('_')[1]) for entry in matching_folders if '_' in entry.name]

        if last_file_number:
            last_file_number = int(np.max(last_file_number))
        else:
            last_file_number = None

        return matching_folders, last_file_number

    def get_result_dir(self) -> pathlib.Path:
        return self.result_path

if __name__ == '__main__':
    result_dir = ResultsPath(
        result_main_dir=config_paths.get('results_path'),
        response_split='ideal'
    )
    result_dir.create_results_dict()
    result_dir.get_result_dir()