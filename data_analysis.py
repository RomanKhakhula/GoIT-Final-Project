# data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_analysis():
  data_path = "D:\\Repos\\Final_GoIT_Project\\internet_service_churn.csv"
  init_data_df = pd.read_csv(data_path)
  init_data_df.set_index(['id'], inplace = True)

  discrete_initial = ['is_tv_subscriber', 'is_movie_package_subscriber', 'service_failure_count', 'download_over_limit']
  continuous_initial = ['subscription_age', 'bill_avg', 'reamining_contract', 'download_avg', 'upload_avg']

  analysis_results = []

  def describe_features(feature, x_dev, dev, x_prob, prob):
    fig = plt.figure(figsize=(15, 3))
    plt.subplot(1, 2, 1)
    plt.bar(x_dev, dev)
    plt.xlabel(feature)
    plt.ylabel('frequency')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_prob, prob, marker='o', markersize=5)
    plt.xlabel(feature)
    plt.ylabel('avg prob of churn')
    plt.grid(True)
    return fig

  def analyse_continuous_features(df, list_of_features, target_column):
    for feature in list_of_features:
        df.sort_values(feature, inplace=True, na_position='first')
        df[f'{feature}_gr'] = -2
        df.loc[df[feature].isna(), f'{feature}_gr'] = -1  # NaN
        df.loc[df[feature] == 0, f'{feature}_gr'] = 0
        rest_shape = df.loc[df[f'{feature}_gr'] == -2].shape[0]
        df.loc[df[f'{feature}_gr'] == -2, f'{feature}_gr'] = [(i // (rest_shape / 10)) + 1 for i in range(rest_shape)]

        description = df[feature].describe()
        analysis_results.append((f'{feature}_desc', description))

        agg_df1 = df.groupby([f'{feature}_gr']).agg({target_column: ['count', 'mean'], feature: ['min', 'max', 'mean']})
        agg_df2 = df.groupby([feature]).agg({target_column: ['count']})
        fig = describe_features(feature, agg_df2.index, agg_df2[target_column, 'count'], agg_df1[feature, 'mean'], agg_df1['churn', 'mean'])
        analysis_results.append((f'{feature}_fig', fig))

  def analyse_discrete_features(df, list_of_features, target_column):
    for feature in list_of_features:
        agg_df1 = df.groupby([feature]).agg({target_column: ['count', 'mean']})
        analysis_results.append((f'{feature}_agg', agg_df1.T))

  analyse_discrete_features(init_data_df, discrete_initial, 'churn')

  work_df = init_data_df.copy()

  for n in range(4):
    work_df[f'service_failure_count_{n}'] = [1 if n == el else 0 for el in list(work_df['service_failure_count'])]
  work_df[f'service_failure_count_4'] = [1 if el >= 4 else 0 for el in list(work_df['service_failure_count'])]

  for n in range(8):
    work_df[f'download_over_limit_{n}'] = [1 if n == el else 0 for el in list(work_df['download_over_limit'])]
  analyse_continuous_features(work_df, continuous_initial, 'churn')

  work_df.loc[work_df['subscription_age'] < 0, 'subscription_age'] = 0
  work_df['reamining_contract'].fillna(0, inplace=True)
  work_df.loc[work_df['reamining_contract'] > 0, 'reamining_contract'] = 1

  for feature in ['download_avg', 'upload_avg']:
    work_df[feature].fillna(work_df[feature].mean(), inplace=True)

  global normalizations_dict
  normalizations_dict = {}
  for feature in ['subscription_age', 'bill_avg', 'download_avg', 'upload_avg']:
    work_df[f'{feature}_norm'] = (work_df[feature] - work_df[feature].min()) / (work_df[feature].max() - work_df[feature].min())
    normalizations_dict[feature] = {'min': work_df[feature].min(), 'max': work_df[feature].max()}

  discrete_additional = [
    'service_failure_count_0', 'service_failure_count_1', 'service_failure_count_2',
    'service_failure_count_3', 'service_failure_count_4', 'download_over_limit_0',
    'download_over_limit_1', 'download_over_limit_2', 'download_over_limit_3',
    'download_over_limit_4', 'download_over_limit_5', 'download_over_limit_6',
    'download_over_limit_7', 'reamining_contract'
    ]
  continuous_additional = [
    'subscription_age_norm', 'bill_avg_norm', 'download_avg_norm', 'upload_avg_norm'
    ]
  analyse_discrete_features(work_df, discrete_additional, 'churn')
  analyse_continuous_features(work_df, continuous_additional, 'churn')

  final_df = work_df[
        ['is_tv_subscriber', 'is_movie_package_subscriber', 'reamining_contract',
         'service_failure_count_0', 'service_failure_count_1', 'service_failure_count_2',
         'service_failure_count_3', 'service_failure_count_4', 'download_over_limit_0',
         'download_over_limit_1', 'download_over_limit_2', 'download_over_limit_3',
         'download_over_limit_4', 'download_over_limit_5', 'download_over_limit_6',
         'download_over_limit_7', 'subscription_age_norm', 'bill_avg_norm',
         'download_avg_norm', 'upload_avg_norm', 'churn']
    ]

  final_df.to_csv('final_df.csv', index=False)
  analysis_results.append(('final_df', final_df.head()))

  return analysis_results, final_df


def plot_correlation_matrix(df, features):
  corr = df[features].corr()
  fig, ax = plt.subplots(figsize=(12, 10)) 
  sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8})
  plt.xticks(rotation=45, ha='right') 
  plt.yticks(rotation=0)
  plt.tight_layout() 
  return fig

def prepare_data_for_prediction(df):
  #df.set_index(['id'], inplace = True)
  for n in range(4):
    df[f'service_failure_count_{n}'] = [1 if n == el  else 0 for el in list(df['service_failure_count'])]
      
  df[f'service_failure_count_4'] = [1 if el >= 4  else 0 for el in list(df['service_failure_count'])]

  for n in range(8):
    df[f'download_over_limit_{n}'] = [1 if n == el  else 0 for el in list(df['download_over_limit'])]
  
  df.loc[df['subscription_age'] < 0, 'subscription_age'] = 0
  df['reamining_contract'].fillna(0, inplace=True)
  df.loc[df['reamining_contract'] > 0, 'reamining_contract'] = 1

  for feature in ['download_avg', 'upload_avg']:
    df[feature].fillna(df[feature].mean(), inplace=True)

  for feature in ['subscription_age', 'bill_avg', 'download_avg', 'upload_avg']:
    df[f'{feature}_norm'] = (df[feature] -  normalizations_dict[feature]['min']) / (normalizations_dict[feature]['max'] - normalizations_dict[feature]['min'])

  
  return df[
        ['id','is_tv_subscriber', 'is_movie_package_subscriber', 'reamining_contract',
         'service_failure_count_0', 'service_failure_count_1', 'service_failure_count_2',
         'service_failure_count_3', 'service_failure_count_4', 'download_over_limit_0',
         'download_over_limit_1', 'download_over_limit_2', 'download_over_limit_3',
         'download_over_limit_4', 'download_over_limit_5', 'download_over_limit_6',
         'download_over_limit_7', 'subscription_age_norm', 'bill_avg_norm',
         'download_avg_norm']
    ]