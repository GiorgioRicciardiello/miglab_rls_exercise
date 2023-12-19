import pathlib
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from config.config import config_paths
from typing import Dict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from numpy import ndarray

class ResponseDictHandler:
    """
     Class that returns the response dict and its inverted version.
    """
    def __init__(self, response_split: Optional[str] = 'ideal'):
        self.response_split = response_split
        # Define the response dict
        if self.response_split == 'ideal':
            # non-reps vs pos vs neg. Both must be removed
            self.response_dict = {
                'positive': 1,
                'both': np.nan,
                'negative': 0,
                'non-responder': 2,
            }

        if self.response_split == 'non_vs_all':
            # non-reps vs (pos & neg & both)
            self.response_dict = {
                'positive': 1,
                'both': 1,
                'negative': 1,
                'non-responder': 0,
            }

        if self.response_split == 'four_class':
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
    def get_encoding_name(self)->str:
        """get the name of the encoding configuration we are using for the target"""
        return self.response_split


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
    plot_class_report = False
    # Balanced Accuracy
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    # F1-Score (Micro-average)
    micro_f1 = f1_score(y_true, y_pred, average='weighted')

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
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # Set the font size for annotations using annot_kws
    sns.heatmap(cm,
                annot=True,
                # fmt="d",
                fmt=".2g",
                cmap="Blues",
                xticklabels=cm_labels,
                yticklabels=cm_labels,
                cbar=True,
                annot_kws={"fontsize": 18},
                vmin =np.min(cm),
                vmax=np.max(cm),
                )
    plt.show()

    if model_name:
        plt.title(f"{model_name}\nAvg F1-Score: {micro_f1:.3f}\nBalanced Acc: {bal_accuracy:.3f}")
    else:
        plt.title(f"Confusion Matrix\nAvg F1-Score: {micro_f1:.3f}\nBalanced Acc: {bal_accuracy:.3f}")

    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()

    return cm, class_report


def generate_results_md(train_set: dict,
                        test_set: dict,
                        model_params: dict,
                        metrics_table: pd.DataFrame,
                        save_path: pathlib.Path,
                        cm_train:pd.DataFrame,
                        cm_test:pd.DataFrame,
                        target_lbls:dict):
    """

    :param train_set: dict, train set {'x': pd.Dataframe, 'y':pd.Dataframe}
    :param test_set: dict, test set {'x': pd.Dataframe, 'y':pd.Dataframe}
    :param model_params: dict, model parameters used to train the model
    :param metrics_table: pd.Dataframe, df of  the metrics in the train and test set we want to publish
    :param save_path: pathlib.Path, path on where to save the md file
    :param cm_train: pd.Dataframe, confusion matrix when prediction is on the train set
    :param cm_test: pd.Dataframe, confusion matrix when prediction is on the test set
    :param target_lbls:dict, target labels {0:'negative', 1:'positive'}
    :return:
        None
    """
    # Format results as Markdown
    results_md = "# Data Splits\n"

    # Train Set
    train_set_info = [
        ("Number of Observations", train_set.get('x').shape[0]),
        ("Number of Features", train_set.get('x').shape[1]),
        ("Count of Unique Values in Target", train_set.get('y').value_counts().to_dict())
    ]

    results_md += "\n### Train Set:\n"
    for label, value in train_set_info:
        if isinstance(value, dict):
            results_md += f"\t{label}:\n"
            for sub_label, sub_value in value.items():
                results_md += f"\t\t{target_lbls.get(sub_label)}: {sub_value}\n"
        else:
            results_md += f"\t{label}: {value}\n"

    # Test Set
    test_set_info = [
        ("Number of Observations", test_set.get('x').shape[0]),
        ("Number of Features", test_set.get('x').shape[1]),
        ("Count of Unique Values in Target", test_set.get('y').value_counts().to_dict())
    ]
    results_md += "\n## Test Set:\n"
    for label, value in test_set_info:
        if isinstance(value, dict):
            results_md += f"\t{label}:\n"
            for sub_label, sub_value in value.items():
                results_md += f"\t\t{target_lbls.get(sub_label)}: {sub_value}\n"
        else:
            results_md += f"\t{label}: {value}\n"

    results_md += "\n# Confusion Matrix:\n"
    # Confusion Matrix for Test Set
    results_md += "\n## Confusion Matrix (Train Set):\n"
    results_md += f"{tabulate(cm_train, headers='keys', tablefmt='github')}\n"


    # Confusion Matrix for Test Set
    results_md += "\n## Confusion Matrix (Test Set):\n"
    results_md += f"{tabulate(cm_test, headers='keys', tablefmt='github')}\n"

    # Metrics Table
    results_md += "\n# Metrics Table:\n"
    results_md += f"{tabulate(metrics_table, headers='keys', tablefmt='github')}\n"

    # Model Parameters
    # params_md = f"\n#Model Parameters:\n\n{tabulate(model_params.items(), tablefmt='github')}\n"
    # params_md = f"\n# Model Parameters:\n\n{model_params}\n"
    params_md = f"\n# Model Parameters:\n\n"
    for key, val in model_params.items():
        params_md += f'\t{key}: {val}\n'


    # Combine results, metrics, and parameters
    md_content = results_md + params_md

    # Save to file
    with open(save_path, "w") as md_file:
        md_file.write(md_content)

    # Print to console
    print(md_content)



def display_cm(y_true: Union[list, np.ndarray],
                            y_pred: Union[list, np.ndarray],
                            cm_labels: dict = None,
                            model_name: Optional[str] = None,
                            save_path= pathlib.Path,
                            figsize: Optional[Tuple] = (8, 6),
                            file_name:str = 'confusion_matrix'
                            ) -> pd.DataFrame:
    """

    :param y_true: (Union[list, np.ndarray]), True labels.
    :param y_pred: (Union[list, np.ndarray]),  Predicted probabilities or scores.
    :param cm_labels: dict, labels to place in the ticks of the cm
            {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    :param model_name: str, name of the model to place in the title

    :return:
        pd.Dataframe, the classification_report
    """
    # Balanced Accuracy
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    # F1-Score (Micro-average)
    micro_f1 = f1_score(y_true, y_pred, average='weighted')

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    # Set the font size for annotations using annot_kws
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                # fmt=".2g",
                cmap="Blues",
                xticklabels=cm_labels,
                yticklabels=cm_labels,
                cbar=False,
                annot_kws={"fontsize": 18},
                vmin=np.min(cm),
                vmax=np.max(cm),
                )

    if model_name:
        plt.title(f"{model_name}\nAvg F1-Score: {micro_f1:.3f}\nBalanced Acc: {bal_accuracy:.3f}")
    else:
        plt.title(f"Confusion Matrix\nAvg F1-Score: {micro_f1:.3f}\nBalanced Acc: {bal_accuracy:.3f}")

    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    cm_df = pd.DataFrame(data=cm, columns=[*cm_labels.keys()],
                         index=[*cm_labels.keys()])
    if save_path:
        save_path.joinpath(file_name+'.xlsx')
        cm_df.to_excel(save_path.joinpath(file_name+'.xlsx'), index=False)
        plt.savefig(save_path.joinpath(file_name+'.png'), dpi=300)
        # if save_path.suffix == '.xlsx':
        #     cm_df.to_excel(save_path, index=False)
        # elif save_path.suffix == '.csv':
        #     cm_df.to_csv(save_path, index=False)
        # else:
        #     print(f'Saving confusion matrix in working path as suffix are not defined')
        #     cm_df.to_excel('cm_model.xlsx', index=False)
    plt.show()
    return cm_df


def grid_search_evaluation_report(model: Any,
                                  scoring: object,
                                  train_set: Dict[str, pd.DataFrame],
                                  test_set: Dict[str, pd.DataFrame],
                                  param_grid: Dict[str, any],
                                  cv: int = 2,
                                  save_path: Optional[pathlib.Path] = None) -> Union[Any, dict]:
    """
     Perform grid search for hyperparameter tuning, train the model, and evaluate its performance.
     Only works for models that have the Sklearn API

    The Grid search is performed only in the given training set. The test set is used to evalaute the best model
    on unseen data.

    :param model: he machine learning model (e.g., classifier or regressor) to be tuned and evaluated.
    :param scoring: Scoring metric used for evaluation during grid search.
    :param train_set: A dictionary containing training set data. Keys: 'x' for features, 'y' for target.
    :param test_set: A dictionary containing test set data. Keys: 'x' for features, 'y' for target.
    :param param_grid:  Dictionary specifying the hyperparameter grid for grid search.
    :param cv: Number of cross-validation folds.
    :param save_path: path to save the excel with the metrics, train and test set
    :return:
         best_classifier: estimator, The trained model with the best hyperparameters.
         best_params: dict, best parameters of the best model
    """

    # Initialize GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(train_set.get('x'), train_set.get('y'))

    # Best hyperparameters
    best_params = grid_search.best_params_
    print(f'Best Parameters: \n{best_params}')

    # Train the model with the best hyperparameters on the entire training set
    best_classifier = grid_search.best_estimator_
    best_classifier.fit(train_set.get('x'), train_set.get('y'))

    # Evaluate the model on the test set
    y_pred_train = best_classifier.predict(train_set.get('x'))
    print(f'Model Report on Train Set:\n{classification_report(train_set.get("y"), y_pred_train)}')

    y_pred_test = best_classifier.predict(test_set.get('x'))
    print(f'Model Report on Test Set:\n{classification_report(test_set.get("y"), y_pred_test)}')

    evaluation_report(train_set=train_set,
                      test_set=test_set,
                      y_pred_train=y_pred_train,
                      y_pred_test=y_pred_test,
                      save_path=save_path)


    return best_classifier, best_params

def evaluation_report(train_set: Dict[str, pd.DataFrame],
                      y_pred_train:ndarray,
                      test_set: Dict[str, pd.DataFrame],
                      y_pred_test:ndarray,
                      target_lbls:dict,
                      save_path:Optional[pathlib.Path] = None) -> pd.DataFrame:
    """
    Function to compute the evaluation report on the observed and predicted observations and save the metrics
    in an Excel file is the save_path is given in the input.

    :param train_set: A dictionary containing training set data. Keys: 'x' for features, 'y' for target.
    :param test_set: A dictionary containing test set data. Keys: 'x' for features, 'y' for target.
    :param y_pred_train: predictions on the training set
    :param y_pred_test: predictions on the test set
    :param save_path:  path to save the excel with the metrics, train and test set
    :return:
        metrics, pd.Dataframe; Metrics in the train and test set
    """

    train_report = classification_report(train_set.get("y"), y_pred_train, output_dict=True)
    test_report = classification_report(test_set.get("y"), y_pred_test, output_dict=True)

    df_train = pd.DataFrame(train_report).transpose()
    df_test = pd.DataFrame(test_report).transpose()

    # Round numerical columns to the third decimal place
    numerical_cols = df_train.select_dtypes(include='number').columns
    df_train[numerical_cols] = df_train[numerical_cols].round(3)
    df_test[numerical_cols] = df_test[numerical_cols].round(3)

    df_train['split'] = 'train'
    df_test['split'] = 'test'
    print(f'Model Report on Train Set:\n{df_train}\n\n')
    print(f'Model Report on Test Set:\n{df_test}')

    # Create DataFrames with the reports
    metrics = pd.concat([df_train, df_test], axis=0)
    # rename the indexes so we have the classes
    new_indexes = []
    for idx in metrics.index:
        try:
            idx_ = idx.split('.')[0]
            idx_ = int(idx_)
            if idx_ in target_lbls.keys():
                new_indexes.append(target_lbls.get(idx_))
            else:
                new_indexes.append(idx)
        except ValueError:
            new_indexes.append(idx)
    metrics.index = new_indexes

    # count of observations on each split
    metrics['ObsCount'] = '-'
    # count the observations we have for each class in each split
    target_count_train = {target_lbls.get(key): val for key, val in train_set.get("y").value_counts().to_dict().items()}
    target_count_test = {target_lbls.get(key): val for key, val in test_set.get("y").value_counts().to_dict().items()}
    # populate the metrics dataframe with the observation count
    for split_ in metrics['split'].unique():
        for class_ in target_lbls.values():
            index_condition = (metrics['split'] == split_) & (metrics.index == class_)
            if split_ == 'train':
                metrics.loc[index_condition, 'ObsCount'] = int(target_count_train.get(class_))
            elif split_ == 'test':
                metrics.loc[index_condition, 'ObsCount'] = int(target_count_test.get(class_))

    if save_path:
        if save_path.suffix == '.xlsx':
            metrics.to_excel(save_path)
            print(f'Metrics saved in path \n {metrics}')
        elif save_path.suffix == '.csv':
            metrics.to_csv(save_path)
            print(f'Metrics saved in path \n {metrics}')
        else:
            print(f'Unable to save metrics. Wong file suffix, available formats are .xlsx and .csv')
    return metrics

def train_val_test_split(x_data, y_data, test_size: float = 0.2, val_size: float = 0.1,
                         random_state: int = 42) -> Tuple:
    """
    Function to split data into training, validation, and test sets.

    :param x_data: Features.
    :param y_data: Target variable.
    :param test_size: The proportion of the dataset to include in the test split.
    :param val_size: The proportion of the dataset to include in the validation split. If None, no validation set is created.
    :param random_state: Seed for random number generation.
    :return: Tuple of arrays (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    if val_size is not None:

        # Split into training and temporary set (80% training + 20% temporary)
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(x_data, y_data,
                                                                      test_size=test_size,
                                                                      random_state=random_state)

        # Further split the temporary set into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp,
                                                          test_size=val_size / (1 - test_size),
                                                          random_state=random_state)

        assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == x_data.shape[0]
        print(f'Data Splits:\n Train: {x_train.shape[0]}\n Val: {x_val.shape[0]}\n Test: {x_test.shape[0]}'
              f'\nTotal: {x_data.shape[0]}')

        return x_train, x_val, x_test, y_train, y_val, y_test

    else:
        # Split only into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size,
                                                            random_state=random_state)
        return x_train, x_test, None, y_train, y_test, None


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

def encode_variables(data, columns,
                     categories=None,
                     encoding_type='ordinal'):
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

def one_hot_encoder_multi_resp_columns(df: pd.DataFrame,
                                       multi_res_column: list,
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
