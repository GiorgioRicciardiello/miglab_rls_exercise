import pandas as pd
from config.config import config_paths
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from utils.utils import identify_feature_types, ResponseDictHandler
import warnings
from tqdm import tqdm
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA, KernelPCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == '__main__':
    random_state = 42
    do_plots = False
    SAVE_PLOTS = True
    SHOW_PLOTS = False
    do_tsne = False
    do_pca = True
    #%%
    # Suppress FutureWarning from Seaborn
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
    #%% Read data
    data = pd.read_csv(config_paths.get('preproc_data_path'))
    #%% target variables
    response_dict_handler = ResponseDictHandler()
    target_response = response_dict_handler.get_response_dict()
    target_response_inv = response_dict_handler.get_inverted_response_dict()
    target = 'response_class'
    #%% plot the target distribution
    target_counts = data[target].value_counts().to_dict()
    df = pd.DataFrame(list(target_counts.items()), columns=['Unique Values', 'Count'])
    df.sort_values(by='Count', ascending=False, inplace=True)
    sns.barplot(x='Unique Values', y='Count', data=df, palette='viridis')
    # for index, row in df.iterrows():
    #     plt.text(index, row['Count'], str(row['Count']), ha='center', va='bottom')
    plt.xlabel('Unique Values')
    plt.ylabel('Count')
    plt.title('Target Distribution')
    plt.grid(axis='y', alpha=0.7)
    plt.xticks(ticks=[*target_response.values()], labels=[*target_response.keys()], rotation=0)
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(config_paths.get('results_dist_path').joinpath('target_distribution.png'),
                    dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.show()

    #%% Identify categorical and continuous features
    feature_types = identify_feature_types(data)
    #%% Plot categorical, continuous and binary data with their respective plots
    if do_plots:
        for feature_type, columns in tqdm(feature_types.items(), desc='Processing Feature Types', unit='type'):
            # feature_type = 'categorical'
            # columns = feature_types.get(feature_type)
            if feature_type == 'binary':
                for col_ in tqdm(columns, desc='Plotting Binary', unit='type'):
                    fig, ax = plt.subplots()
                    data.pivot_table(index=[target], columns=col_, aggfunc='size').plot(kind='bar',
                                                                                        legend=True,
                                                                                        title=col_,
                                                                                        ax=ax)
                    if col_ == 'dem_sex':
                        labels = ['Female', 'Male']
                        handles, labels = plt.gca().get_legend_handles_labels()
                        ax.legend(handles, labels, title='Gender')
                    ax.grid(axis='y', alpha=0.7)
                    ax.set_xticks(ticks=[*target_response.values()], labels=[*target_response.keys()], rotation=0)
                    ax.set_ylabel(ylabel='Count')
                    plt.tight_layout()
                    if SAVE_PLOTS:
                        plt.savefig(config_paths.get('results_dist_path').joinpath('binary', f'{col_}.png'),
                                    dpi=300)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.clf()  # Clear the current figure
                    plt.cla()  # Clear the current axis

            elif feature_type == 'continuous':
                xticks = [x + 1 for x in [*target_response.values()]]
                for col_ in tqdm(columns, desc='Plotting Continuous', unit='type'):
                    fig, ax = plt.subplots()
                    sns.boxplot(data=data, x=target, y=col_, ax=ax)
                    ax.set_xticks(ticks=[*target_response.values()], labels=[*target_response.keys()], rotation=0)
                    ax.grid(axis='y', alpha=0.7)
                    plt.tight_layout()
                    if SAVE_PLOTS:
                        plt.savefig(config_paths.get('results_dist_path').joinpath('continuous', f'{col_}.png'),
                                    dpi=300)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.clf()  # Clear the current figure
                    plt.cla()  # Clear the current axis

            elif feature_type == 'categorical':
                for col_ in tqdm(columns, desc='Plotting Categorical', unit='type'):
                    fig, ax = plt.subplots()
                    data.pivot_table(index=[target], columns=col_, aggfunc='size').plot(kind='bar',
                                                                                        legend=True,
                                                                                        title=col_,
                                                                                        ax=ax)
                    ax.grid(axis='y', alpha=0.7)
                    ax.set_xticks(ticks=[*target_response.values()], labels=[*target_response.keys()], rotation=0)
                    ax.set_ylabel(ylabel='Count')
                    plt.tight_layout()
                    if SAVE_PLOTS:
                        plt.savefig(config_paths.get('results_dist_path').joinpath('categorical', f'{col_}.png'),
                                    dpi=300)
                    if SHOW_PLOTS:
                        plt.show()
                    plt.clf()  # Clear the current figure
                    plt.cla()  # Clear the current axis

            else:
                print(f'Undefined type {feature_type}')

    y_data = data[target].copy()
    x_data = data.drop(columns=[target, 'responseid']).copy()
    #%% Standardize the dataset
    feature_types.get('categorical').remove(target)
    column_transformer = ColumnTransformer([
        ('standard_scaler_continuous', StandardScaler(), feature_types.get('continuous')),  # Z-transform continuous
        ('standard_scaler_categorical', MinMaxScaler(), feature_types.get('categorical')),  # Min-Max categorical
        ('passthrough_binary', 'passthrough', feature_types.get('binary')),  # pass binary
        # ('passthrough_responseid', 'passthrough', 'responseid')
    ])
    column_order = feature_types.get('continuous') +feature_types.get('categorical') +  feature_types.get('binary')

    x_data_proc = column_transformer.fit_transform(x_data)
    x_data_proc = pd.DataFrame(x_data_proc, columns=column_order)
    x_data_proc.describe().T


    # %% TNSE for data visualization
    if do_tsne:
        tsne = TSNE(n_components=6,
                    perplexity=40,
                    early_exaggeration=12.0,
                    learning_rate="auto",
                    n_iter=800,
                    n_iter_without_progress=300,
                    min_grad_norm=1e-7,
                    metric="euclidean",
                    metric_params=None,
                    init="pca",
                    verbose=2,
                    random_state=random_state,
                    method="exact",
                    angle=0.5,
                    n_jobs=None, )
        X_tsne = tsne.fit_transform(x_data)

        tsne.kl_divergence_
        tsne_df = pd.DataFrame(data=X_tsne, columns=[f'Component {comp}' for comp in range(0, X_tsne.shape[1])])
        # tsne_df['y_true'] = y_true
        # Plot the t-SNE results
        marker_size = 50
        scatter_kws = {'edgecolors': 'k'}

        scat_mx = scatter_matrix(frame=tsne_df,
                       alpha=0.8,
                       diagonal='kde',
                       c=y_data,
                       cmap='viridis',
                       figsize=(20, 20),
                       marker='o',
                       s=50,
                       **scatter_kws)
        # # Create a separate legend using dummy points
        # legend_handles = [plt.Line2D([0], [0],
        #                              marker='o',
        #                              color='w',
        #                              markerfacecolor=sns.color_palette('viridis')[i],
        #                              markersize=10, label=label) for i, label in target_response_inv.items()]
        #
        # plt.legend(handles=legend_handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.suptitle('Scatter Matrix of t-SNE Components') #, y=0.94)
        # Customize the scatter plots in the matrix
        for ax in scat_mx.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=12, rotation=0)
            ax.set_ylabel(ax.get_ylabel(), fontsize=12, rotation=90)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.imshow([np.unique(y_data)])
        _ = plt.xticks(ticks=np.unique(y_data), labels=target_response)
        plt.show()

    if do_pca:
        pca = PCA(n_components=6)
        kernel_pca = KernelPCA(
            n_components=6,
            kernel="rbf",
            gamma=10,
            fit_inverse_transform=True,
            alpha=0.1
        )
        marker_size = 50
        scatter_kws = {'edgecolors': 'k'}

        kpca_data = kernel_pca.fit_transform(x_data_proc)
        kpca_df = pd.DataFrame(data=kpca_data, columns=[f'Component {comp}' for comp in range(0, kpca_data.shape[1])])


        kpca_mx = scatter_matrix(frame=kpca_df,
                       alpha=0.8,
                       diagonal='kde',
                       c=y_data,
                       cmap='viridis',
                       figsize=(20, 20),
                       marker='o',
                       s=50,
                       **scatter_kws)
        plt.show()

        pca_data = pca.fit_transform(x_data_proc)
        pca_df = pd.DataFrame(data=pca_data, columns=[f'Component {comp}' for comp in range(0, pca_data.shape[1])])

        pca.explained_variance_
        pca.explained_variance_ratio_
        pca.singular_values_

        pca_mx = scatter_matrix(frame=pca_df,
                                 alpha=0.8,
                                 diagonal='kde',
                                 c=y_data,
                                 cmap='viridis',
                                 figsize=(20, 20),
                                 marker='o',
                                 s=50,
                                 **scatter_kws)
        plt.show()
