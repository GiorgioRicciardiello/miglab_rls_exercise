import pandas as pd
from config.config import config_paths
from matplotlib import pyplot as plt
from utils.utils import identify_feature_types, ResponseDictHandler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from utils.utils import evaluate_classification
import operator

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
    X_train_proc_df.describe().T

    # Transform the test set using the fitted pipeline
    X_test_proc = preprocessing_pipeline.transform(X_test)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=column_order)
    X_test_proc_df.describe().T


    #%% model testing

    #%% XGboost
    if logistic_model == 'xgboost':
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

    # %% ElasticNet classifier
    if logistic_model == 'elastic_net':
        sgd_model = SGDClassifier(
            max_iter=1000,
            penalty='elasticnet',
            loss='log_loss',
            alpha=0.1,
            l1_ratio=0.15,
            fit_intercept=True,
            tol=1e-3,
            shuffle=False,
            verbose=50,
            random_state=random_state,
            learning_rate='adaptive',
            eta0=0.001,
            power_t=0.5,
            early_stopping=False,
            class_weight='balanced')
        sgd_model.fit(X_train, y_train)
        # Make predictions on the test set
        y_pred_train = sgd_model.predict(X_train)
        y_pred_test = sgd_model.predict(X_test)

        report_train = evaluate_classification(y_true=y_train.values,
                                               y_pred=y_pred_train,
                                               cm_labels=target_response,
                                               report_classes_names=target_response_inv,
                                               model_name='SGDClassifier - Train')

        report_test = evaluate_classification(y_true=y_test.values,
                                              y_pred=y_pred_test,
                                              cm_labels=target_response,
                                              report_classes_names=target_response_inv,
                                              model_name='SGDClassifier - Test')

        # plot_model_coefficients(model=sgd_model,
        #                         trained_features=X_train,
        #                         top_n=10,
        #                         model_name='sgd_model'
        #                         )
        #
        # plot_model_coefficients(model=sgd_model,
        #                         trained_features=X_train,
        #                         top_n=20,
        #                         model_name='Elastic Net',
        #                         figsize=(10, 8),
        #                         )
    # %% Simple Logistic Regression
    # if logistic_model == 'simple_logistic':
    #     X_train_with_constant = sm.add_constant(X_train)  # Add a constant term to the features
    #     ols_model = sm.OLS(y_train, X_train_with_constant).fit()
    #     print(ols_model.summary())
    #     y_pred_train_ols = ols_model.predict(X_train_with_constant)
    #     X_test_with_constant = sm.add_constant(X_test)  # Add a constant term to the test features
    #     y_pred_test_ols = ols_model.predict(X_test_with_constant)
    #
    #     # evaluate the predictions
    #     evaluate_classification(y_true=y_pred_train_ols,
    #                             y_pred=y_pred_train,
    #                             cm_labels=target_response,
    #                             model_name='OLS model - Train',
    #                             )
    #     # valuate the testing
    #     evaluate_classification(y_true=y_pred_test_ols,
    #                             y_pred=y_pred_test,
    #                             cm_labels=target_response,
    #                             model_name='OLS model - Test',
    #                             )


    # %% DecisionTreeClassifier
    if logistic_model == 'DecisionTreeClassifier':
        clf = DecisionTreeClassifier(random_state=random_state,
                                     max_depth=8,
                                     min_samples_split=4,
                                     min_samples_leaf=2,
                                     class_weight='balanced'
                                     )
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)


        f1_train = f1_score(y_train, y_pred_train, average='macro')  # or 'micro', 'weighted', etc.
        f1_test = f1_score(y_test, y_pred_test, average='macro')  # or 'micro', 'weighted', etc.

        # if we want to do cross validation
        # f1_scorer = make_scorer(f1_score, average='macro')  # or 'micro', 'weighted', etc.
        # # Perform cross-validation with F1 score as the scoring metric
        # cross_val_scores = cross_val_score(estimator=clf,
        #                                    X=X_train,
        #                                    y=y_train,
        #                                    cv=5,
        #                                    scoring=f1_scorer
        #                                    )



    # %% Ordinal Regression
    if logistic_model == 'ordinal_regression':
        pass











