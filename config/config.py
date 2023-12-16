"""
Author: Giorgio Ricciardiello
        giocrm@stanford.edu
configurations parameters for the paths
"""
import pathlib

config_paths = {
    'root_path': pathlib.Path(__file__).parents[1],
    'raw_data_path':  pathlib.Path(__file__).parents[1].joinpath(pathlib.Path(r'data/raw_data/rls_data.csv')),
    'preproc_data_path':  pathlib.Path(__file__).parents[1].joinpath(
        pathlib.Path(r'data/pre_process/pre_process_data.csv')),
    'results_path':  pathlib.Path(__file__).parents[1].joinpath(pathlib.Path(r'results')),
    'results_dist_path': pathlib.Path(__file__).parents[1].joinpath(pathlib.Path(r'results/distributions')),
}
