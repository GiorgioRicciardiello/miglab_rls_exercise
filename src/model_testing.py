import pandas as pd
import pathlib
from config.config import config_paths
from typing import Union
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from matplotlib import pyplot as plt
import seaborn as sns
from src.utils import identify_feature_types, ResponseDictHandler
import warnings
from tqdm import tqdm
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score
from src.utils import evaluate_classification
from sklearn.model_selection import cross_val_score
import operator

if __name__ == '__main__':
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
    X_train_proc_df.describe().T

    # Transform the test set using the fitted pipeline
    X_test_proc = preprocessing_pipeline.transform(X_test)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=column_order)
    X_test_proc_df.describe().T


    #%% model testing

    #%% XGboost
    params = {
        'objective': 'multi:softmax',
        'num_class': len(y_data.unique()),
        'eta': 0.01,  # Adjust the learning rate
        'max_depth': 4,  # Adjust the maximum depth
        'reg_alpha': 0.4,  # L1 norm - prevent overfitting - pushing some of the coefficients toward zero
        'reg_lambda': 0.3,  # L2 norm - penalizing the sum of squared weights.
        'gamma': 0.2,  # penalty for creating new nodes in the tree - controlling the complexity
        'subsample': 0.7,  # Adjust subsampling
        'colsample_bytree': 0.7,  # Adjust feature subsampling
        'device': 'cuda',
    }

    eval_results = {}
    watchlist = [(xgb.DMatrix(X_train, label=y_train), 'train')]
    xgb_model = xgb.train(
        params,
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=3000,
        evals=watchlist,
        evals_result=eval_results,
        verbose_eval=50,
    )

    # train the model
    dtrain = xgb.DMatrix(X_train)
    y_pred_train = xgb_model.predict(dtrain)
    # test the model
    dtest = xgb.DMatrix(X_test)
    y_pred_test = xgb_model.predict(dtest)

    # evaluate the predictions
    evaluate_classification(y_true=y_train,
                            y_pred=y_pred_train,
                            cm_labels=target_response,
                            model_name='XGBoost - Train',
                            )
    # valuate the testing
    evaluate_classification(y_true=y_test,
                            y_pred=y_pred_test,
                            cm_labels=target_response,
                            model_name='XGBoost - Test',
                            )

    # evaluate the model
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, xgb_model.num_boosted_rounds() + 1),
             eval_results['train']['mlogloss'],
             label='Training Error',
             color='orange')
    plt.title('XGBoost Training Curve')
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel('Training Error')
    plt.grid(0.7)
    plt.legend()
    plt.show()

    # Get the feature importance score and plot each importance type in the same figure
    num_imp_features = 15
    importance_types = {'weight': {}, 'gain': {}, 'cover': {}, 'total_gain': {}, 'total_cover': {}}
    best_gs_features = set()
    for imp_ in importance_types.keys():
        imp_importance = xgb_model.get_score(importance_type=imp_, fmap='')
        importance_types[imp_] = dict(sorted(imp_importance.items(), key=operator.itemgetter(1), reverse=True))
        best_gs_features.update([*importance_types[imp_].keys()][0:num_imp_features])
    importance_features_df = pd.DataFrame(importance_types)

    fig, axes = plt.subplots(nrows=len(importance_types), ncols=1, figsize=(10, 5 * len(importance_types)))
    colors = ['skyblue', 'salmon', 'darkgreen', 'orange', 'purple']
    for idx, importance_type in enumerate(importance_features_df):
        top_features = importance_features_df.iloc[:, idx].sort_values(ascending=False).head(num_imp_features)
        top_features.plot(kind='barh', ax=axes[idx], color=colors[idx], legend=True)
        axes[idx].set_title(f'Feature Importance ({importance_type.capitalize()})')
    plt.tight_layout()
    plt.show()

    #%% Sklearn - Ordinal Regression

    #%%traditional ordinal regression
    X_train_with_constant = sm.add_constant(X_train)  # Add a constant term to the features
    ols_model = sm.OLS(y_train, X_train_with_constant).fit()
    print(ols_model.summary())
    y_pred_train_ols = ols_model.predict(X_train_with_constant)
    X_test_with_constant = sm.add_constant(X_test)  # Add a constant term to the test features
    y_pred_test_ols = ols_model.predict(X_test_with_constant)

    # evaluate the predictions
    evaluate_classification(y_true=y_pred_train_ols,
                            y_pred=y_pred_train,
                            cm_labels=target_response,
                            model_name='OLS model - Train',
                            )
    # valuate the testing
    evaluate_classification(y_true=y_pred_test_ols,
                            y_pred=y_pred_test,
                            cm_labels=target_response,
                            model_name='OLS model - Test',
                            )

    # DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=random_state)
    cross_val_score(estimator=clf,
                    X=X_train,
                    y=y_train,
                    cv=5
                    )



    #%%
    classifiers_pipeline = Pipeline([
        # ('preprocessing', column_transformer),
        # ('xgboost', XGBClassifier()),  # XGBoost classifier
        ('ordinal_regression', sm.OLS(y_train, sm.add_constant(X_train))),  # Replace with the actual ordinal regression model
        ('decision_tree', DecisionTreeClassifier()),  # Decision Tree classifier
        ('random_forest', RandomForestClassifier())  # Random Forest classifier
    ])

    # Fit the pipeline on the training data
    classifiers_pipeline.fit(X_train, y_train)

    # Predictions on the test set
    y_pred_xgboost = classifiers_pipeline.predict(X_test, 'xgboost')
    y_pred_ordinal_regression = classifiers_pipeline.predict(X_test, 'ordinal_regression')
    y_pred_decision_tree = classifiers_pipeline.predict(X_test, 'decision_tree')
    y_pred_random_forest = classifiers_pipeline.predict(X_test, 'random_forest')

    # Calculate accuracy on the test set
    accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)
    accuracy_ordinal_regression = accuracy_score(y_test, y_pred_ordinal_regression)
    accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
    accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)

    print(f'Accuracy - XGBoost: {accuracy_xgboost}')
    print(f'Accuracy - Ordinal Regression: {accuracy_ordinal_regression}')
    print(f'Accuracy - Decision Tree: {accuracy_decision_tree}')
    print(f'Accuracy - Random Forest: {accuracy_random_forest}')

    # Calculate F1 score on the test set
    f1_score_xgboost = f1_score(y_test, y_pred_xgboost, average='macro')  # or 'micro', 'weighted', etc.
    f1_score_ordinal_regression = f1_score(y_test, y_pred_ordinal_regression, average='macro')
    f1_score_decision_tree = f1_score(y_test, y_pred_decision_tree, average='macro')
    f1_score_random_forest = f1_score(y_test, y_pred_random_forest, average='macro')

    print(f'F1 Score - XGBoost: {f1_score_xgboost}')
    print(f'F1 Score - Ordinal Regression: {f1_score_ordinal_regression}')
    print(f'F1 Score - Decision Tree: {f1_score_decision_tree}')
    print(f'F1 Score - Random Forest: {f1_score_random_forest}')















