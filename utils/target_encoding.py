"""
Class used to encode the target response
"""
from typing import Union, Optional, Any
import numpy as np
import math

class TargetEncoding:
    """
     Class that returns the response dict and its inverted version.
    """
    def __init__(self, response_split: Optional[str] = 'ideal'):
        options = ['PosVsNeg', 'ideal', 'NonVsAll']
        self.response_split = response_split
        # Define the response dict
        # primary analysis
        if self.response_split == 'PosVsNeg':
            self.response_dict = {
                'positive': 1,
                'both': np.nan,  # these observations are remove
                'negative': 0,
                'non-responder': np.nan,  # these observations are remove
            }
            # self._remove_nan_values()

        elif self.response_split == 'ideal':
            # non-reps vs pos vs neg. Both must be removed
            self.response_dict = {
                'positive': 1,
                'both': np.nan,  # these observations are remove
                'negative': 0,
                'non-responder': 2,
            }
            # self._remove_nan_values()


        elif self.response_split == 'NonVsAll':
            # non-reps vs (pos & neg & both)
            self.response_dict = {
                'positive': 1,
                'both': 1,
                'negative': 1,
                'non-responder': 0,
            }
            # self._remove_nan_values()


        # if self.response_split == 'four_class':
        #     # Define the response dict
        #     self.response_dict = {
        #         'positive': 1,
        #         'both': 2,
        #         'negative': 3,
        #         'non-responder': 0,
        #     }
        #     self._remove_nan_values()

        else:
            raise ValueError('Please define an available target encoding method\n'
                             'The options are:\n'
                             f"\t {options}")

    def merge_keys_with_same_values(self):
        # Reverse the dictionary to have values as keys and lists of original keys as values
        reversed_dict = {}
        for key, value in self.response_dict.items():
            reversed_dict.setdefault(value, []).append(key)

        # Merge keys with the same values
        merged_dict = {}
        for merged_key, original_keys in reversed_dict.items():
            merged_dict["_".join(original_keys)] = merged_key

        self.response_dict = merged_dict


    def remove_nan_values(self):
        """Remove the nan values, so we do not have them in the labels, as we remove this observations"""
        self.response_dict = {key: value for key, value in self.response_dict.items() if not math.isnan(value)}

    def get_lbls(self)->dict:
        """Return the label dictionary were the keys are integers and the values the names"""
        return dict(sorted(self.response_dict.items(), key=lambda item: item[1]))

    def get_inv_lbls(self)->dict:
        """Return the inverted label dictionary were the keys are the names and the values the numbers"""
        inverted_response_dict = {v: k for k, v in sorted(self.response_dict.items())}
        return dict(sorted(inverted_response_dict.items()))

    def get_encoding_name(self)->str:
        """get the name of the encoding configuration we are using for the target"""
        return self.response_split