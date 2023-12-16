from typing import Dict, List, Union
import pandas as pd
from typing import Union, Optional
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)

import matplotlib.pyplot as plt
import seaborn as sns

class ResponseDictHandler:
    """
     Class that returns the response dict and its inverted version.
    """
    def __init__(self):
        # Define the response dict
        self.response_dict = {
            'positive': 1,
            'both': 2,
            'negative': 3,
            'non-responder': 0,
        }

    def get_response_dict(self)->dict:
        # Return the response dict
        return dict(sorted(self.response_dict.items(), key=lambda item: item[1]))

    def get_inverted_response_dict(self)->dict:
        # Invert the response dict
        inverted_response_dict = {v: k for k, v in sorted(self.response_dict.items())}
        return dict(sorted(inverted_response_dict.items()))


def evaluate_classification(y_true: Union[list, np.ndarray],
                            y_pred: Union[list, np.ndarray],
                            cm_labels: dict = None,
                            model_name: Optional[str] = None,
                            report_classes_names: Optional[dict] = None,
                            ) -> pd.DataFrame:
    """

    :param y_true: (Union[list, np.ndarray]), True labels.
    :param y_pred: (Union[list, np.ndarray]),  Predicted probabilities or scores.
    :param cm_labels: dict, labels to place in the ticks of the cm
            {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    :param model_name: str, name of the model to place in the title
    :param report_classes_names: labels to rename the columns of the classes in the classification_report call
            {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}

    :return:
        pd.Dataframe, the classification_report
    """
    # df_results = pd.DataFrame(y_true.values, columns=['y_true'])
    # df_results['y_pred'] = y_pred
    # fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    # # ROC for each class, separately
    # for i, value in enumerate(range(0, num_classes)):
    #     subset_df = df_results.copy()
    #     # Binarize 'y_true' for the current class
    #     y_true_bin = label_binarize(subset_df['y_true'], classes=[value])
    #     # Create a one-vs-all classifier
    #     classifier = OneVsRestClassifier(model_with_constraints)
    #     # Fit the classifier
    #     classifier.fit(subset_df.drop('y_true', axis=1), y_true_bin)
    #     # Predict probabilities for the current class
    #     y_score = classifier.predict_proba(subset_df.drop('y_true', axis=1))
    #     # Compute ROC curve and AUC
    #     fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, 1])  # Use y_score[:, 1] for the positive class
    #     roc_auc = auc(fpr, tpr)
    #     # Plot ROC curve on the i-th subplot
    #     axes[i].plot(fpr, tpr, label=f'Class {value} (AUC = {roc_auc:.2f})')
    #     axes[i].plot([0, 1], [0, 1], 'k--')
    #     axes[i].set_title(f'ROC Curve - Class {value}')
    #     axes[i].set_xlabel('False Positive Rate')
    #     axes[i].set_ylabel('True Positive Rate')
    #     axes[i].legend()
    #
    # plt.show()

    plot_class_report = False
    # Balanced Accuracy
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Classification Report
    class_report = classification_report(y_true=y_true,
                                         y_pred=y_pred,
                                         output_dict=True)
    class_report = pd.DataFrame(class_report)
    if report_classes_names:
        class_report.rename(columns=report_classes_names, inplace=True)
    # Remove the 'support' row
    class_report.drop('support', axis=0, inplace=True)

    # plot the class report
    if plot_class_report:
        class_report_rounded = class_report.round(2)
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(class_report_rounded,
                              annot=True,
                              cmap='Blues',
                              fmt=".2f",
                              linewidths=.5,
                              cbar=False,
                              linecolor='black'
                              )
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Classes')
        ax.set_title('Classification Report Heatmap')
        plt.tight_layout()
        plt.show()

    # Display Metrics
    print("Classification Report:")
    print(class_report)

    # Confusion Matrix
    # F1-Score (Micro-average)
    micro_f1 = f1_score(y_true, y_pred, average='weighted')

    # Confusion Matrix
    if cm_labels is None:
        cm_labels = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, cbar=False)
    if model_name:
        plt.title(f"{model_name}\nAvg F1-Score: {micro_f1:.3f}\nBalanced Acc: {bal_accuracy:.3f}")
    else:
        plt.title(f"Confusion Matrix\nAvg F1-Score: {micro_f1:.3f}\nBalanced Acc: {bal_accuracy:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return class_report


def identify_feature_types(data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify categorical, continuous, and binary features in a DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.

    Returns:
    - Dict[str, List[str]]: Dictionary containing lists of column names for each feature type.
      Keys: 'binary', 'continuous', 'categorical'.
    """
    # Define binary values
    bin_values = [0, 1, -1]

    # Dictionary to store column names for each feature type
    column_types: Dict[str, List[str]] = {'binary': [], 'continuous': [], 'categorical': []}

    # Iterate through each column in the DataFrame
    for col_name, col in data.items():
        # Check the type of each column
        if col.apply(lambda x: isinstance(x, str)).any():
            # Skip columns with string values
            continue

        unique_values = col.unique()

        # Check if the column is binary
        if np.all(np.isin(unique_values, bin_values)):
            column_types['binary'].append(col_name)

        # Check if the column is categorical (based on the number of unique values)
        elif len(unique_values) < 5:
            column_types['categorical'].append(col_name)

        # If neither binary nor categorical, consider it continuous
        else:
            column_types['continuous'].append(col_name)

    return column_types

def rls_func(x, numerical: bool = True) -> Union[str, int]:
    """Recode the numerical rls severity into severity ordinal categories."""
    rls_groups = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Very Severe'}

    if x == 0:
        return rls_groups.get(0) if not numerical else 0
    if 0 < x <= 11:
        return rls_groups.get(1) if not numerical else 1
    if 11 < x <= 21:
        return rls_groups.get(2) if not numerical else 2
    if 21 < x <= 31:
        return rls_groups.get(3) if not numerical else 3
    else:
        return rls_groups.get(4) if not numerical else 4

def encode_variables(data, columns, categories=None, encoding_type='ordinal'):
    """
    Encode variables in a DataFrame or Series.

    Parameters:
    - data: DataFrame or Series
    - columns: List of columns to encode
    - categories: List of category lists for ordinal encoding (required for 'ordinal' encoding type)
    - encoding_type: Type of encoding ('ordinal', 'ordinal_unordered', or 'onehot')

    Returns:
    - DataFrame or Series with encoded variables
    """
    if encoding_type == 'ordinal':
        if categories is None:
            raise ValueError("Categories must be provided for 'ordinal' encoding type.")
        encoder = OrdinalEncoder(categories=categories)
    elif encoding_type == 'ordinal_unordered':
        encoder = OrdinalEncoder()
    elif encoding_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore')
    else:
        raise ValueError("Invalid encoding_type. Use 'ordinal', 'ordinal_unordered', or 'onehot'.")

    if isinstance(data, pd.DataFrame):
        data[columns] = encoder.fit_transform(data[columns])
    elif isinstance(data, pd.Series):
        data = encoder.fit_transform(data.values.reshape(-1, 1))
    else:
        raise ValueError("Input data must be a DataFrame or Series.")

    return data

def flatten_nested_list(nested_list:list[list])-> list:
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list

def remap_unknown_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remaps the -55 values in categorical columns to the next available positive integer
    that is not already present in the column. Skips columns with string values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with -55 values replaced in the specified categorical columns.
    """
    for col in df.columns:
        # Identify unique non-null values in the column
        unique_values = df[col].dropna().unique()

        # skip column that contain strings
        for uniq_ in unique_values:
            if isinstance(uniq_, str):
                continue

        # Skip columns with no -55 values
        if -55 not in unique_values:
            continue
        # replace the -55 with the next positive value available
        next_positive_int = np.max(np.sort(unique_values)) + 1
        # Replace -55 with the next available positive integer
        df[col] = df[col].replace(-55, next_positive_int)
    return df

def identify_multiple_response_columns(df:pd.DataFrame) -> list:
    """
    Multiple response column contain a coma to separate the answers, we will identify them
    :param df: dataframe, dataset to identify the comas
    :return: list, columns names where a coma was found
    """
    columns_with_commas = []
    # Iterate over columns
    for column in df.columns:
        # Get unique values, ignoring NaN
        unique_values = df[column].dropna().unique()
        # Check if any unique value contains a comma
        contains_comma = any(',' in str(value) for value in unique_values)

        # If commas are found, add the column to the list
        if contains_comma:
            columns_with_commas.append(column)

    return columns_with_commas

def one_hot_encoder_multi_resp_columns(df: pd.DataFrame, multi_res_column: list,
                                       inplace: bool = False) -> pd.DataFrame:
    """
    Columns that contain multiple response answer and teh responses are separated by comas. We perform a one-hot
    encoding to the column and remove the original one
    :param df: pd.DataFrame, dataset to be modified
    :param multi_res_column: list, names of columns that are of multiple response and coma separated in a str
    :param inplace: TODO not working, set it to False
    :return:
    """
    if not inplace:
        df = df.copy()

    initial_shape = df.shape[0]

    def encode_responses(row, mapping_dict):
        if isinstance(row, str):
            responses = row.split(',')
            for response in responses:
                mapping_dict[response] = 1
        return mapping_dict

    for col_ in multi_res_column:
        mapping_dict = {}  # create an empty mapping dictionary
        mapping_dict = df[col_].apply(lambda row: encode_responses(row, mapping_dict)).iloc[-1]

        dummy_df = df[col_].str.get_dummies(sep=',').rename(lambda x: col_ + '_' + x, axis=1)
        dummy_df = dummy_df.rename(columns=lambda x: x.lower().replace('-', '_').replace(' ', '_'))

        # Merge dummy_df with the original DataFrame
        df = pd.merge(df, dummy_df, left_index=True, right_index=True, how='outer')

    # Drop the original columns after merging
    df.drop(columns=multi_res_column, inplace=True)

    # Sanity check of unchanged rows
    assert initial_shape == df.shape[0]

    return df if not inplace else None
