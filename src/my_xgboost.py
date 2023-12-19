import pathlib

import numpy as np
import pandas as pd
from config.config import config_paths
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import f1_score, make_scorer
import operator
from utils.utils import (identify_feature_types, evaluate_classification, evaluation_report, display_cm,
                         generate_results_md)
from sklearn.decomposition import PCA
from tabulate import tabulate
import joblib
from utils.results_path import ResultsPath
from utils.target_encoding import TargetEncoding
from numpy import ndarray

if __name__ == '__main__':
    random_state = 42
    target_method = 'PosVsNeg'  #  ['PosVsNeg', 'ideal', 'NonVsAll']
    target = 'response_class'
    # %% Read data
    data = pd.read_csv(config_paths.get('preproc_data_path'))
    # %% target Encoding and filtering
    target_encoding = TargetEncoding(response_split=target_method)
    data[target] = data[target].map(target_encoding.get_lbls())
    # Drop rows with NaN values in the 'response_class' column
    data.dropna(subset=target, inplace=True)
    data.reset_index(inplace=True, drop=True)
    # remove nan values as they will not be used for the labels
    target_encoding.remove_nan_values()
    target_encoding.merge_keys_with_same_values()
    assert len(data[target].unique()) == len([*target_encoding.get_lbls().keys()])
    # %% Create the path for the results
    result_dir = ResultsPath(
        result_main_dir=config_paths.get('results_path'),
        response_split=target_method
    )
    result_dir.create_results_dict()
    result_path = result_dir.get_result_dir()  # in this folder we will save all the metrics for the current model
    # %% Identify categorical and continuous features
    feature_types = identify_feature_types(data)
    # %% remove more features
    strings_to_remove = ['rls_sfdq13_1', 'rls_sfdq13_2', 'rls_sfdq13_3', 'rls_sfdq13_4', 'rls_sfdq13_9b',
                         'rls_screen_withouttod', 'rls_screen_withtod']

    filtered_categorical = [col for col in feature_types.get('categorical', []) if 'rls_sfdq13_' not in col]
    feature_types['binary'] = [col for col in feature_types.get('binary', []) if col not in strings_to_remove]
    feature_types.get('binary').remove('rls_diagnosis')

    # Create a DataFrame from the feature_types dictionary
    df_feature_types = pd.DataFrame([(key, val) for key, values in feature_types.items() for val in values],
                                    columns=['feature_type', 'feature_name'])

    df_feature_types.to_excel(result_path.joinpath('FeatureList_XGBoost.xlsx'), index=False)

    # Drop rows with NaN values in the 'response_class' column
    # %% Split features and target
    y_data = data[target].copy()
    x_data = data.drop(columns=[target, 'responseid']).copy()
    # %% Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.2,
                                                        random_state=random_state)

    # Further split the temporary set into training and validation sets (50% temporary + 50% validation)
    # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # X_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x_data=x_data,
    #                                                                      y_data=y_data,
    #                                                                      test_size=0.2,
    #                                                                      val_size=0.1,
    #                                                                      random_state=random_state)

    # Get the number of observations and features for each split
    num_obs_train, num_features_train = X_train.shape
    num_obs_test, num_features_test = X_test.shape

    # Count of unique values in the target variable for train and test sets
    target_counts_train = y_train.value_counts().to_dict()
    target_counts_test = y_test.value_counts().to_dict()

    # Display the results
    print("\nTrain Set:")
    print(f"\tNumber of Observations: {num_obs_train}")
    print(f"\tNumber of Features: {num_features_train}")
    print("\tCount of Unique Values in Target:")
    for label, count in target_counts_train.items():
        print(f"\t\t{label}: {count}")

    print("\nTest Set:")
    print(f"\tNumber of Observations: {num_obs_test}")
    print(f"\tNumber of Features: {num_features_test}")
    print("\tCount of Unique Values in Target:")
    for label, count in target_counts_test.items():
        print(f"\t\t{label}: {count}")

    # %% Standardize the dataset and normalize both feature splits
    for feature_type_key, feature_type_val in feature_types.items():
        if target in feature_type_val:
            feature_types.get(feature_type_key).remove(target)
            print(f'\nTarget removed from type {feature_type_key}\n')

    # feature_types.get('categorical').remove(target)
    column_transformer = ColumnTransformer([
        ('standard_scaler_continuous', StandardScaler(), feature_types.get('continuous')),  # Z-transform continuous
        ('standard_scaler_categorical', MinMaxScaler(), feature_types.get('categorical')),  # Min-Max categorical
        ('passthrough_binary', 'passthrough', feature_types.get('binary')),  # pass binary
        # ('passthrough_responseid', 'passthrough', 'responseid')
    ])
    column_order = feature_types.get('continuous') + feature_types.get('categorical') + feature_types.get('binary')

    # Create the pipeline
    preprocessing_pipeline = Pipeline([
        ('column_transformer', column_transformer),
    ])

    # Fit and transform the training set
    X_train_proc = preprocessing_pipeline.fit_transform(X_train)
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=column_order)

    # Fit and transform the test set
    X_test_proc = preprocessing_pipeline.transform(X_test)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=column_order)

    # Group them in a single variable
    train_set = {'x': X_train_proc_df, 'y': y_train}
    test_set = {'x': X_test_proc_df, 'y': y_test}

    # Define the F1 score as the scoring metric
    scorer = make_scorer(f1_score, average='weighted')


    # %% Xgboost model

    # %% Find best hyperparameters
    # xgb_classifier = xgb.XGBClassifier(verbose_eval=500,
    #                                    device='cuda',
    #                                    objective='multi:softmax',
    #                                    num_class=len(y_data.unique()),
    #                                    num_boost_round=3000,
    #                                    )
    # # Define the hyperparameters grid for grid search
    # param_grid = {
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'n_estimators': [20, 30, 50],
    #     'subsample': [0.7, 1.0],
    #     'colsample_bytree': [0.7, 1.0],
    #     'gamma': [0, 1, 5],
    #     'reg_alpha': [0.1, 0.5, 1.0],
    #     'reg_lambda': [0.1, 0.5, 1.0],
    # }
    #
    # best_classifier, best_params = grid_search_evaluation_report(model=xgb_classifier,
    #                                                              param_grid=param_grid,
    #                                                              cv=2,
    #                                                              scoring=scorer,
    #                                                              train_set=train_set,
    #                                                              test_set=test_set,
    #                                                              )
    #
    # y_pred_train = best_classifier.predict(train_set.get('x'))
    # y_pred_test = best_classifier.predict(test_set.get('x'))
    #
    # evaluation_report(train_set=train_set,
    #                   test_set=test_set,
    #                   y_pred_train=y_pred_train,
    #                   y_pred_test=y_pred_test,
    #                   # save_path=config_paths.get('results_path').joinpath('Metrics_LogisticRegression.xlsx')
    #                   save_path=None
    #                   )

    # %% Run a single model
    params = {
        'objective': 'multi:softmax',
        'num_class': len(y_data.unique()),
        'n_estimators': 20,
        'eta': 0.01,  # Adjust the learning rate  for long boosting iterations 0.001
        'max_depth': 6,  # Adjust the maximum depth
        'reg_alpha': 2,  # Increase L1 norm for stronger regularization
        'reg_lambda': 1.3,  # Increase L2 norm for stronger regularization
        'gamma': 0.6,  # Increase penalty for creating new nodes for stronger regularization
        'subsample': 0.6,  # Adjust subsampling
        'colsample_bytree': 0.6,  # Adjust feature subsampling
        'device': 'cuda',
    }
    eval_results = {}
    watchlist = [(xgb.DMatrix(train_set.get('x'), label=train_set.get('y')), 'train')]
    xgb_model = xgb.train(
        params,
        xgb.DMatrix(train_set.get('x'), label=train_set.get('y')),
        num_boost_round=60000,
        evals=watchlist,
        evals_result=eval_results,
        verbose_eval=200,
    )

    # Prediction - train
    dtrain = xgb.DMatrix(train_set.get('x'))
    y_pred_train = xgb_model.predict(dtrain)
    # Predictions - test
    dtest = xgb.DMatrix(test_set.get('x'))
    y_pred_test = xgb_model.predict(dtest)

    metrics = evaluation_report(train_set=train_set,
                                test_set=test_set,
                                y_pred_train=y_pred_train,
                                y_pred_test=y_pred_test,
                                save_path=result_path.joinpath('Metrics_XGBoost.xlsx'),
                                # save_path=None
                                target_lbls=target_encoding.get_inv_lbls()
                                )
    # metrics_tab = tabulate(metrics, headers='keys', tablefmt='pretty', showindex=False)

    # Save the trained model
    model_filename = 'xgb_model.bin'
    xgb_model.save_model(result_path.joinpath(model_filename))
    # Save the model as a pickle file (optional)
    joblib.dump(xgb_model, result_path.joinpath('xgb_model.pkl'))

    # evaluate the predictions - if we want to plot the confusion matrix
    cm_train = display_cm(y_true=train_set.get('y'),
                            y_pred=y_pred_train,
                            cm_labels=target_encoding.get_lbls(),
                            model_name=f'XGBoost - Train - Target Encoding {target_method}',
                            save_path=result_path,
                            file_name = 'Metric_CM_Train',
                            )

    # valuate the testing
    cm_test = display_cm(y_true=test_set.get('y'),
                            y_pred=y_pred_test,
                            cm_labels=target_encoding.get_lbls(),
                            model_name=f'XGBoost - Test - Target Encoding {target_method}',
                            save_path=result_path,
                            file_name = 'Metric_CM_Test',
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
    plt.savefig(result_path.joinpath('XGBoostTrainingLoss.png'), dpi=300)
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
    importance_features_df.to_excel(result_path.joinpath('XGBoostFeatureImportanceTable.xlsx'), index=True)

    # plot in a single figure
    fig, axes = plt.subplots(nrows=len(importance_types), ncols=1, figsize=(10, 5 * len(importance_types)))
    colors = ['skyblue', 'salmon', 'darkgreen', 'orange', 'purple']
    for idx, importance_type in enumerate(importance_features_df):
        top_features = importance_features_df.iloc[:, idx].sort_values(ascending=False).head(num_imp_features)
        top_features.plot(kind='barh', ax=axes[idx], color=colors[idx], legend=True)
        axes[idx].set_title(f'Feature Importance ({importance_type.capitalize()})')
    plt.tight_layout()
    plt.grid(0.7)
    plt.savefig(result_path.joinpath('XGBoostFeatureImportance.png'), dpi=300)
    plt.show()

    # plot in multiple figures
    colors = ['skyblue', 'salmon', 'darkgreen', 'orange', 'purple']
    for idx, importance_type in enumerate(importance_features_df):
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = importance_features_df.iloc[:, idx].sort_values(ascending=False)
        top_features.plot(kind='barh', ax=ax, color=colors[idx], legend=True)
        ax.set_title(f'Feature Importance ({importance_type.capitalize()})')
        plt.tight_layout()
        plt.grid(0.7)
        plt.savefig(result_path.joinpath(f'XGBoostFeatureImportance_{importance_type}.png'), dpi=300)
        plt.show()


    # Example usage:
    # Call this function with your actual data and parameters
    generate_results_md(train_set=train_set,
                        test_set=test_set,
                        model_params=params,
                        save_path=result_path.joinpath('ExperimentSummary.md'),
                        metrics_table=metrics,
                        cm_train=cm_train,
                        cm_test=cm_test,
                        target_lbls=target_encoding.get_inv_lbls()
                        )



    # %%
    #
    # pca = PCA(n_components=6)
    # kernel_pca = KernelPCA(
    #     n_components=6,
    #     kernel="rbf",
    #     gamma=10,
    #     fit_inverse_transform=True,
    #     alpha=0.1
    # )
    # marker_size = 50
    # scatter_kws = {'edgecolors': 'k'}
    #
    # kpca_data = kernel_pca.fit_transform(train_set.get('x'))
    #
    #
    # kpca_df = pd.DataFrame(data=kpca_data, columns=[f'Component {comp}' for comp in range(0, kpca_data.shape[1])])
    #
    # kpca_mx = scatter_matrix(frame=kpca_df,
    #                          alpha=0.8,
    #                          diagonal='kde',
    #                          c=train_set.get('y'),
    #                          cmap='viridis',
    #                          figsize=(20, 20),
    #                          marker='o',
    #                          s=10,
    #                          **scatter_kws)
    # plt.show()
    #
    # pca_data = pca.fit_transform(train_set.get('x'))
    # pca_df = pd.DataFrame(data=pca_data, columns=[f'Component {comp}' for comp in range(0, pca_data.shape[1])])
    #
    # pca.explained_variance_  # It shows how much variance is captured by each principal component.
    # pca.explained_variance_ratio_
    # pca.singular_values_
    #
    # pca_mx = scatter_matrix(frame=pca_df,
    #                         alpha=0.8,
    #                         diagonal='kde',
    #                         c=train_set.get('y'),
    #                         cmap='viridis',
    #                         figsize=(40, 40),
    #                         marker='o',
    #                         s=50,
    #                         **scatter_kws)
    # plt.show()
    #
    # pca_df['Target'] = train_set.get('y')
    # # Scatter plot of the first two principal components
    # sns.scatterplot(x='Component 1', y='Component 2', hue='Target', data=pca_df)
    # plt.title('PCA Scatter Plot of the First Two Components')
    # plt.show()

    # %% XGBoost with PCA
    # pca = PCA(n_components=6)
    # pca_train = pca.fit_transform(train_set.get('x'))
    # pca_train = pd.DataFrame(data=pca_train, columns=[f'Component {comp}' for comp in range(1, pca_data.shape[1] + 1)])
    # pca_train_set = {'x': pca_train, 'y': train_set.get('y')}
    #
    # pca_test = pca.fit_transform(test_set.get('x'))
    # pca_test = pd.DataFrame(data=pca_test, columns=[f'Component {comp}' for comp in range(1, pca_data.shape[1] + 1)])
    # pca_test_set = {'x': pca_test, 'y': test_set.get('y')}
    #
    # params = {
    #     'objective': 'multi:softmax',
    #     'num_class': len(y_data.unique()),
    #     'n_estimators': 20,
    #     'eta': 0.001,  # Adjust the learning rate
    #     'max_depth': 6,  # Adjust the maximum depth
    #     'reg_alpha': 2,  # Increase L1 norm for stronger regularization
    #     'reg_lambda': 1.3,  # Increase L2 norm for stronger regularization
    #     'gamma': 0.6,  # Increase penalty for creating new nodes for stronger regularization
    #     'subsample': 0.6,  # Adjust subsampling
    #     'colsample_bytree': 0.6,  # Adjust feature subsampling
    #     'device': 'cuda',
    # }
    # eval_results = {}
    # watchlist = [(xgb.DMatrix(pca_train_set.get('x'), label=train_set.get('y')), 'train')]
    # xgb_model = xgb.train(
    #     params,
    #     xgb.DMatrix(pca_train_set.get('x'), label=train_set.get('y')),
    #     num_boost_round=15000,
    #     evals=watchlist,
    #     evals_result=eval_results,
    #     verbose_eval=50,
    # )
    #
    # # train the model
    # dtrain = xgb.DMatrix(pca_train_set.get('x'))
    # y_pred_train = xgb_model.predict(dtrain)
    # # test the model
    # dtest = xgb.DMatrix(pca_test_set.get('x'))
    # y_pred_test = xgb_model.predict(dtest)
    #
    # evaluation_report(train_set=pca_train_set,
    #                   test_set=pca_test_set,
    #                   y_pred_train=y_pred_train,
    #                   y_pred_test=y_pred_test,
    #                   # save_path=config_paths.get('results_path').joinpath('Metrics_XGBoost.xlsx')
    #                   save_path=None
    #                   )
    #
    # pca_feature = {'bmi',
    #                'dem_age',
    #                'dem_sex',
    #                'height_m',
    #                'ipaq_2',
    #                'ipaq_4',
    #                'ipaq_6',
    #                'ipaq_sit_min',
    #                'ipaq_total',
    #                'plm_presence',
    #                'rls_duration_1',
    #                'rls_med_frequency',
    #                'rls_peak_hour_1',
    #                'rls_sfdq13_5',
    #                'rls_sfdq13_6_night',
    #                'rls_sfdq13_7_mid_day',
    #                'secondary_conditions',
    #                'sirls_1',
    #                'sirls_10',
    #                'sirls_2',
    #                'sirls_3',
    #                'sirls_5',
    #                'sirls_9',
    #                'weight_kg'}
