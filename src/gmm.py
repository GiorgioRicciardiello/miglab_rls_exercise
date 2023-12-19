import pandas as pd
from config.config import config_paths
from utils.utils import identify_feature_types, ResponseDictHandler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score

if __name__ == '__main__':
    #%% Input
    logistic_model = 'SGDClassifier'  # which classification model we want to test
    #%%
    random_state = 42
    #%% Read data
    data = pd.read_csv(config_paths.get('preproc_data_path'))
    #%% target variables
    response_dict_handler = ResponseDictHandler()
    target_response = response_dict_handler.get_response_dict()
    target_response_inv = response_dict_handler.get_inverted_response_dict()
    target = 'response_class'
    #%% Identify categorical and continuous features
    feature_types = identify_feature_types(data)
    #%% Split features and target
    y_data = data[target].copy()
    x_data = data.drop(columns=[target, 'responseid']).copy()
    #%% Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.2,
                                                        random_state=random_state)

    #%% Standardize the dataset and normalize both feature splits
    feature_types.get('categorical').remove(target)
    column_transformer = ColumnTransformer([
        ('standard_scaler_continuous', StandardScaler(), feature_types.get('continuous')),  # Z-transform continuous
        ('standard_scaler_categorical', MinMaxScaler(), feature_types.get('categorical')),  # Min-Max categorical
        ('passthrough_binary', 'passthrough', feature_types.get('binary')),  # pass binary
        # ('passthrough_responseid', 'passthrough', 'responseid')
    ])
    column_order = feature_types.get('continuous') +feature_types.get('categorical') +  feature_types.get('binary')

    # Create the pipeline
    preprocessing_pipeline = Pipeline([
        ('column_transformer', column_transformer),
    ])

    # Fit and transform the training set
    X_train_proc = preprocessing_pipeline.fit_transform(X_train)
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=column_order)

    # Transform the test set using the fitted pipeline
    X_test_proc = preprocessing_pipeline.transform(X_test)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=column_order)

    # Group them in a single variable
    train_set = {'x': X_train_proc_df, 'y': y_train}
    test_set = {'x': X_test_proc_df, 'y': y_test}

    # Define the F1 score as the scoring metric
    scorer = make_scorer(f1_score, average='weighted')

    #%%