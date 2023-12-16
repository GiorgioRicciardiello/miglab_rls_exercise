import pandas as pd
from config.config import config_paths
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import math
from src.utils import (rls_func, encode_variables, flatten_nested_list, remap_unknown_category,
                       identify_multiple_response_columns, one_hot_encoder_multi_resp_columns,
                       ResponseDictHandler)

if __name__ == '__main__':
    target = 'response_class'  # ordinal, engineered
    choices = ['positive', 'negative', 'both', 'non-responder']
    # %% Read data
    raw_data = pd.read_csv(config_paths.get('raw_data_path'))
    print(f'Data initial dimensions: {raw_data.shape}')
    # %% set all column names to lower case, replace - for _
    raw_data.columns = raw_data.columns.str.lower().str.replace('-', '_')
    #%% Drop rows where all values are NaN, ignoring 'responder' and 'nonresponder' columns
    raw_data = raw_data.dropna(subset=raw_data.columns.difference(['responder', 'nonresponder']), how='all')
    #%% re-map do not know responses
    raw_data:pd.DataFrame = remap_unknown_category(df=raw_data)
    #%% define the target
    response_dict_handler = ResponseDictHandler()
    # Drop rows where either 'pos_exp_better_cat' or 'neg_exp_worse_cat' contains NaN values
    raw_data = raw_data.dropna(subset=['pos_exp_better_cat', 'neg_exp_worse_cat'])
    conditions = [
        raw_data['pos_exp_better_cat'].isin([1, 2]) & raw_data['neg_exp_worse_cat'].eq(0),  # positive responders
        raw_data['pos_exp_better_cat'].eq(0) & raw_data['neg_exp_worse_cat'].isin([1, 2]),  # negative responders
        raw_data['pos_exp_better_cat'].isin([1, 2]) & raw_data['neg_exp_worse_cat'].isin([1, 2]),
        # BOTH positive and negative responders
        raw_data['pos_exp_better_cat'].eq(0) & raw_data['neg_exp_worse_cat'].eq(0),  # non-responders
    ]
    raw_data[target] = np.select(conditions, choices, default=0)
    resp_df_ = raw_data.where(raw_data.response_class != '0').dropna(how='all')
    resp_df_ = resp_df_.where(raw_data.response_class!='both').dropna(how='all')
    # remove the experience columns as they were used to make the target
    exp_columns = [col for col in raw_data.columns if '_exp_' in col ]
    exp_columns.extend(['nonresponder', 'responder'])
    raw_data.drop(columns=exp_columns, inplace=True)
    # make the target numeric
    # Get the response dict
    response_dict = response_dict_handler.get_response_dict()
    unique_values = raw_data['response_class'].unique()
    missing_values = set(unique_values) - set(response_dict.keys())
    if not missing_values:
        print("All unique values are present in response_dict.")
    else:
        print(f"Missing values: {missing_values}")
    raw_data['response_class'] = raw_data['response_class'].map(response_dict)

    #%% Handling outliers and possible errors in data imputation
    open_question_columns = {
        'rls_diag_year': 'How old were you (in years) when you were diagnosed with restless legs syndrome (RLS)?' ,
        'rls_diag_physician': 'What kind of doctor gave you the diagnosis of RLS? ',
        'dem_weight': 'Weight (lbs)',
        'rls_duration_1': 'How many years have you been experiencing symptoms of RLS?' ,
        'rls_med_dose': 'How much of the medication do you take? (Example: 300 mg)',
        'ipaq_2':'How many minutes did you usually spend doing vigorous physical activities on one of those days?',
        'ipaq_4': 'How many minutes did you usually spend doing moderate physical activities on one of those days?',
        'ipaq_6': 'How many minutes did you usually spend walking on one of those days?',
        'ipaq_7_1': 'During the last 7 days, how much time did you spend sitting on a week day?_hours',
        'ipaq_7_3': 'During the last 7 days, how much time did you spend sitting on a week day?_minutes',
    }

    # rls_diag_year
    # In rls_diag_year some wronlgy wrote the year of the diagnosis but not the age they had at the time of diagnosis
    # Identify rows where 'rls_diag_year' is greater than 'dem_age'
    incorrect_diagnosis_rows = raw_data['rls_diag_year'] > raw_data['dem_age']
    # Calculate birth year using 'dem_age'
    birth_year = pd.to_datetime(raw_data['startdate']).dt.year - raw_data['dem_age']
    # Calculate age at diagnosis for incorrect diagnoses
    raw_data.loc[incorrect_diagnosis_rows, 'rls_diag_year'] = (
            pd.to_datetime(raw_data.loc[incorrect_diagnosis_rows, 'rls_diag_year'], format='%Y').dt.year - birth_year[incorrect_diagnosis_rows]
    )

    # dem_weight
    # weight that are zero -_-
    # Replace zero values in 'dem_weight' with the mean based on matching 'dem_height_1' and 'dem_height_2'
    raw_data['dem_weight'] = raw_data['dem_weight'].replace(0, pd.NA)
    raw_data['dem_weight'] = raw_data.groupby(['dem_height_1', 'dem_height_2'])['dem_weight'].transform(
        lambda x: x.fillna(x.mean()))
    raw_data['dem_weight'] = raw_data['dem_weight'].astype(int)

    # rls_duration_1
    # rls_duration_1 How many years have you been experiencing symptoms of RLS?
    if (raw_data['rls_duration_1'] > raw_data['dem_age']).any() and not (
            pd.isna(raw_data['rls_duration_1']) | pd.isna(raw_data['dem_age'])).any():
        raise ValueError("Condition True: Patients that have RLS duration > age. Approximate Smartly")

    # ipaq_7_1
    # A week has max 168 hours, it cannot surpass this number
    if (raw_data['ipaq_7_1'] >= 168).any():
        # Calculate the mean excluding values greater than or equal to 168
        mean_excluding_outliers = raw_data.loc[raw_data['ipaq_7_1'] < 168, 'ipaq_7_1'].mean()
        raw_data.loc[raw_data['ipaq_7_1'] >= 168, 'ipaq_7_1'] = mean_excluding_outliers

    # ipaq_7_3
    # A week has max 10080 minutes, it cannot surpass this number
    if (raw_data['ipaq_7_3'] >= 10080).any():
        mean_excluding_outliers = raw_data.loc[raw_data['ipaq_7_3'] < 10080, 'ipaq_7_3'].mean()
        raw_data.loc[raw_data['ipaq_7_3'] >= 10080, 'ipaq_7_3'] = mean_excluding_outliers

    # %% drop unwanted columns
    unwanted_columns = ['progress', 'startdate', 'enddate', 'status', 'duration (in seconds)', 'finished',
                        'recordeddate', 'distributionchannel', 'userlanguage',  'consent', 'rls_diag_physician',
                        'rls_diag_physician_recode', 'rls_med_responsive', 'rls_med_name', 'rls_med_dose',
                        'rls_med_timing']
    more_unwanted_columns = [col for col in raw_data.columns if 'essay' in col or
                           'ex_' in col or
                           '_why' in col or
                           'program_' in col]
    unwanted_columns = unwanted_columns + more_unwanted_columns
    print(f'Unwanted columns to remove: {len(unwanted_columns)}\n {unwanted_columns}')
    raw_data.drop(columns=unwanted_columns, inplace=True)
    print(f'New dimension after column removal {raw_data.shape}')

    #%% Categorical encoding in the features
    categorical_features = ['rls_bilateral', 'rls_med_frequency', 'rls_sfdq13_3']
    # Create a DataFrame with the selected categorical features
    categorical_data = raw_data[categorical_features]
    encoder = OrdinalEncoder()
    encoded_data = encoder.fit_transform(categorical_data)
    # Replace the original features with the encoded features in the original DataFrame
    raw_data[categorical_features] = encoded_data
    #%% One-Hot-Encoding to multiple response features
    multi_res_cols:list = identify_multiple_response_columns(df=raw_data)
    raw_data:pd.DataFrame = one_hot_encoder_multi_resp_columns(df=raw_data,
                                                               multi_res_column=multi_res_cols,
                                                               inplace=False)
    # %% rls experience columns
    rls_experience_col = [col for col in raw_data.columns if 'exp' in col]
    print(f'Missing values in the RLS experience columns: \n {raw_data[rls_experience_col].isna().sum()}')
    # %% data engineering - Define and Assess RLS Characteristics
    sirls_col = [sirls for sirls in raw_data.columns if 'sirls' in sirls]
    raw_data['rls_severity'] = raw_data.loc[:, sirls_col].sum(axis=1)
    raw_data.drop(columns=sirls_col, inplace=True)  # maybe not drop
    raw_data.rls_severity.describe()
    raw_data.rls_severity = raw_data.rls_severity.apply(rls_func, numerical=True)
    # rls_pregnancy
    rls_pregnancy = [rls_p for rls_p in raw_data.columns if 'rls_pregnancy' in rls_p]
    # Create the new 'pregnancy' column
    raw_data['rls_pregnancy'] = np.where(
        ((raw_data[rls_pregnancy[0]].notna() & (raw_data[rls_pregnancy[0]] != 0)) |
         (raw_data[rls_pregnancy[1]].notna() & (raw_data[rls_pregnancy[1]] != 0))),
        1,  # Set to 1 if any of the conditions is met
        np.where(
            (raw_data[rls_pregnancy].notna().any(axis=1)),
            0,  # Set to 0 if both columns are NaN
            np.nan  # Set to NaN if both columns are NaN
        )
    )
    raw_data.drop(columns=rls_pregnancy, inplace=True)
    # IPAQ Physical Activity Scoring
    raw_data.ipaq_2 = raw_data.ipaq_2.abs()
    raw_data.ipaq_2.values[raw_data.ipaq_2 > 240] = 240
    raw_data.ipaq_2 = raw_data.ipaq_2.apply(lambda x: x if x < 240 else 240)

    raw_data.ipaq_4 = raw_data.ipaq_4.abs().astype(float)
    raw_data.ipaq_4.values[raw_data.ipaq_4 > 240] = 240

    raw_data['ipaq_mod'] = 4 * raw_data.ipaq_3_19  # *df['ipaq_4a']
    raw_data.drop(columns='ipaq_3_19', inplace=True)
    raw_data['ipaq_vig'] = 8 * raw_data.ipaq_1_4
    raw_data.drop(columns='ipaq_1_4', inplace=True)

    raw_data.ipaq_6 = raw_data.ipaq_6.abs()
    raw_data.ipaq_6.values[raw_data.ipaq_6 > 240] = 240

    raw_data['ipaq_walk'] = 3.3 * raw_data.ipaq_5_1  # *df['IPAQ_6a']

    raw_data['ipaq_total'] = raw_data.ipaq_vig + raw_data.ipaq_mod + raw_data.ipaq_walk
    raw_data.drop(columns=['ipaq_vig','ipaq_mod', 'ipaq_walk', 'ipaq_5_1'], inplace=True)

    # IPAQ Sedentary Time
    ipaq_columns = [col for col in raw_data.columns if 'ipaq_' in col]
    # All values should be positive expect for the no response coding
    raw_data[ipaq_columns] = raw_data[ipaq_columns].map(
        lambda val: np.abs(val) if isinstance(val, (int, float)) and not math.isnan(val) and val != -55 else 0
    )
    raw_data[ipaq_columns] = raw_data[ipaq_columns].astype(int)
    # Update the typing
    for col in ipaq_columns:
        raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce', downcast='integer')

    # raw_data['ipaq_sit_min'] = (raw_data.ipaq_7_1 * 60) + raw_data.ipaq_7_3   # hours to minutes then add minutes
    # raw_data.drop(columns=['ipaq_7_1','ipaq_7_3'], inplace=True)
    #
    # raw_data['ipaq_sit_hrs'] = raw_data.ipaq_sit_min/ 60
    # raw_data.ipaq_sit_hrs.where(raw_data.ipaq_sit_hrs < 24, np.nan, inplace=True)

    #%% demographics
    raw_data['dem_age'] = raw_data['dem_age'].astype(int)

    raw_data.dem_sex = raw_data.dem_sex.map({'Male': 0, 'Female': 1, 'Prefer not to answer':2})
    # Filter rows where 'dem_sex' values are either 0 or 1
    raw_data = raw_data[(raw_data['dem_sex'] == 0) | (raw_data['dem_sex'] == 1)]
    # race is almost the same for all, not useful, then remove
    raw_data.drop(columns='dem_race', inplace=True)
    # create BMI and remove associate variables
    raw_data = (
        raw_data.assign(
            height_m=lambda x: (x['dem_height_1'] * 0.3048 + x['dem_height_2'] * 0.0254).clip(1, 2),
            weight_kg=lambda x: x['dem_weight'] / 2.205,
            bmi=lambda x: round(x['dem_weight'] / (x['height_m'] ** 2), 2)
        )
        .loc[lambda x: x['bmi'] >= 10]  # Keep rows where BMI is greater than or equal to 10
        .drop(columns=['dem_height_1', 'dem_height_2', 'dem_weight'])
    )

    #%% Handle missing values
    raw_data.fillna(-1, inplace=True)
    #%% constant columns
    # bin_values = [0, 1, -1]
    # for col_name, col in raw_data.items():
    #     col_name = [*raw_data.columns][1]
    #     col = raw_data[col_name]
    #
    #     if col.apply(lambda x: isinstance(x, str)).any():
    #         continue
    #     unique_values = col.unique()
    #     if np.all(np.isin(unique_values, [bin_values])):
    #


    #%% Save the pre-process dataset
    raw_data.to_csv(config_paths.get('preproc_data_path'),index=False)
    print(f'Dimension of resulting pre-process dataset {raw_data.shape}')


