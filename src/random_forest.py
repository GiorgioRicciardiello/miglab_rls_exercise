import pandas as pd
from config.config import config_paths
from matplotlib import pyplot as plt
from utils.utils import identify_feature_types, ResponseDictHandler, grid_search_evaluation_report, evaluation_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

if __name__ == '__main__':
    # %% Input
    logistic_model = 'SGDClassifier'  # which classification model we want to test
    # %%
    random_state = 42
    # %% Read data
    data = pd.read_csv(config_paths.get('preproc_data_path'))
    # %% target variables
    response_dict_handler = ResponseDictHandler()
    target_response = response_dict_handler.get_response_dict()
    target_response_inv = response_dict_handler.get_inverted_response_dict()
    target = 'response_class'
    # %% Identify categorical and continuous features
    feature_types = identify_feature_types(data)
    # %% Split features and target
    y_data = data[target].copy()
    x_data = data.drop(columns=[target, 'responseid']).copy()
    # %% Split training and test set
    X_train, X_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=0.2,
                                                        random_state=random_state)

    # %% Standardize the dataset and normalize both feature splits
    feature_types.get('categorical').remove(target)
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

    # Transform the test set using the fitted pipeline
    X_test_proc = preprocessing_pipeline.transform(X_test)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=column_order)

    # Group them in a single variable
    train_set = {'x': X_train_proc_df, 'y': y_train}
    test_set = {'x': X_test_proc_df, 'y': y_test}

    # Define the F1 score as the scoring metric
    scorer = make_scorer(f1_score, average='weighted')

    # %% classifier
    classifier = RandomForestClassifier(n_estimators=100,
                                        criterion='log_loss',
                                        random_state=42,
                                        verbose=100,
                                        class_weight='balanced')

    # Hyperparameter tuning with GridSearchCV and cross-validation
    param_grid = {
        'n_estimators': [10, 20, 30, 50],
        'max_depth': [2, 4, 5, 6, 8, 10],
        'min_samples_split': [2, 5, 7, 10],
        'min_samples_leaf': [1, 2, 4, 6],
        'ccp_alpha': [0.010, 0.035, 0.050, 0.01, 0.03],
    }

    best_classifier, best_params = grid_search_evaluation_report(model=classifier,
                                                                 param_grid=param_grid,
                                                                 cv=2,
                                                                 scoring=scorer,
                                                                 train_set=train_set,
                                                                 test_set=test_set,
                                                                 )
    # Fit the model with the best parameters
    rfc_best = RandomForestClassifier(random_state=random_state,
                                      criterion='log_loss',
                                      verbose=100,
                                      class_weight='balanced',
                                      **best_params)

    rfc_best.fit(train_set.get('x'), train_set.get('y'))

    y_pred_train = rfc_best.predict(train_set.get('x'))
    y_pred_test = rfc_best.predict(test_set.get('x'))

    evaluation_report(train_set=train_set,
                      test_set=test_set,
                      y_pred_train=y_pred_train,
                      y_pred_test=y_pred_test,
                      save_path=config_paths.get('results_path').joinpath('Metrics_RandomForestClassifier.xlsx'))

    # The higher, the more important the feature. Gini importance.
    classifier.feature_importances_

    feat_imp_df = pd.DataFrame({'Feature': [*train_set.get('x').columns],
                                'NormalizedGiniImpurityGain': classifier.feature_importances_, })

    feat_imp_df = feat_imp_df.sort_values(by='NormalizedGiniImpurityGain', ascending=False)
    mean = feat_imp_df.loc[:, 'NormalizedGiniImpurityGain'].mean()
    feat_imp_df['hue'] = np.where(
        feat_imp_df['NormalizedGiniImpurityGain'] < feat_imp_df.loc[:, 'NormalizedGiniImpurityGain'].mean(),
        'skyblue', 'tomato')

    # Plot the coefficients
    plt.figure(figsize=(10, 28))
    plt.barh(feat_imp_df['Feature'], feat_imp_df['NormalizedGiniImpurityGain'], color=feat_imp_df['hue'])
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Feature Importance RandomForestClassifier')
    # Customize plot appearance
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(config_paths.get('results_path').joinpath('RandomForestFeatureImportance.png'), dpi=300)
    plt.show()

    # %% Decision Tree Classifier
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
    # Total impurity of leaves vs effective alphas of pruned tree
    clf = DecisionTreeClassifier(random_state=random_state,
                                 class_weight='balanced')
    # Create a sample weight array with the same shape as y_train
    sample_weights = np.ones_like(y_train)

    path = clf.cost_complexity_pruning_path(X=X_train,
                                            y=y_train,
                                            )
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # plot the alphas
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()

    # train a decision tree using the effective alphas
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=random_state,
                                     ccp_alpha=ccp_alpha)
        clf.fit(train_set.get('x'), train_set.get('y'))
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    # show that the number of nodes and tree depth decreases as alpha increases.
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.show()

    train_scores = [clf.score(train_set.get('x'), train_set.get('y')) for clf in clfs]
    test_scores = [clf.score(test_set.get('x'), test_set.get('y')) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
