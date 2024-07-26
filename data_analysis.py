# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

init_data_df = pd.read_csv('internet_service_churn.csv')
init_data_df.set_index(['id'], inplace = True)
# print(init_data_df.head())

discrete_initial = ['is_tv_subscriber', 'is_movie_package_subscriber', 'service_failure_count', 'download_over_limit']
continuous_initial = ['subscription_age', 'bill_avg', 'reamining_contract', 'download_avg', 'upload_avg']

def describe_features(feature, x_dev, dev, x_prob, prob):
  fig = plt.figure(figsize=(15, 3))

  plt.subplot(1, 2, 1)
  plt.bar(x_dev, dev)
  plt.xlabel(feature)
  plt.ylabel('frequency')
  plt.grid(True)

  plt.subplot(1, 2, 2)
  plt.plot(x_prob, prob, marker = 'o', markersize = 5)
  plt.xlabel(feature)
  plt.ylabel('avg prob of churn')
  plt.grid(True)

  plt.show()

def analyse_discrete_features(df: pd.DataFrame, list_of_features: list, target_column: chr):
  for feature in list_of_features:
    print(f'{feature} (describe)' + ' *'*50)
    print(pd.DataFrame([list(df[feature].describe().values) + [(df[feature].isnull().sum() / df.shape[0])]], columns = list(df[feature].describe().index) + ['part of NaN'], index = None))
    print('\n')

    print(f'{feature} (frequency / avg prob of {target_column})' + ' *'*50)
    agg_df = df.groupby([feature]).agg({target_column: ['count', 'mean']})
    print(agg_df.T)
    print('\n')

    describe_features(feature, agg_df.index, agg_df[target_column, 'count'], agg_df.index, agg_df[target_column, 'mean'])
    print('\n'*2)

def analyse_continuous_features(df: pd.DataFrame, list_of_features: list, target_column: chr):
  for feature in list_of_features:
    df.sort_values(feature, inplace = True, na_position = 'first')
    df[f'{feature}_gr'] = -2
    df.loc[df[feature].isna(), f'{feature}_gr'] = -1 #NaN
    df.loc[df[feature] == 0, f'{feature}_gr'] = 0
    rest_shape = df.loc[df[f'{feature}_gr'] == -2].shape[0]
    df.loc[df[f'{feature}_gr'] == -2, f'{feature}_gr'] = [(i // (rest_shape / 10)) + 1 for i in range(rest_shape)]

    print(f'{feature} (describe)' + ' *'*50)
    print(pd.DataFrame([list(df[feature].describe().values) + [(df[feature].isnull().sum() / df.shape[0])]], columns = list(df[feature].describe().index) + ['part of NaN'], index = None))
    print('\n')

    print(f'{feature} (frequency / avg prob of {target_column})' + ' *'*50)
    agg_df1 = df.groupby([f'{feature}_gr']).agg({target_column: ['count', 'mean'], feature: ['min','max','mean']})
    agg_df2 = df.groupby([feature]).agg({target_column: ['count']})
    print(agg_df1.T)
    print('\n')

    describe_features(feature, agg_df2.index, agg_df2[target_column, 'count'], agg_df1[feature, 'mean'], agg_df1['churn', 'mean'])
    print('\n'*2)

# is_tv_subscriber, is_movie_package_subscriber - без змін
# service_failure_count - групуємо 4+, OneHotEncod-имо
# download_over_limit - OneHotEncod-имо
work_df = init_data_df

for n in range(4):
  work_df[f'service_failure_count_{n}'] = [1 if n == el  else 0 for el in list(work_df['service_failure_count'])]
work_df[f'service_failure_count_4'] = [1 if el >= 4  else 0 for el in list(work_df['service_failure_count'])]

for n in range(8):
  work_df[f'download_over_limit_{n}'] = [1 if n == el  else 0 for el in list(work_df['download_over_limit'])]

# subscription_age - відємні значення в 0, нормалізуємо
# bill_avg - нормалізуємо
# reamining_contract - NaN в 0, перетворюємо в категоріальну(0 / 1)
# download_avg, upload_avg - Nan замінюємо на середні значення, нормалізуємо
work_df.loc[work_df['subscription_age'] < 0, 'subscription_age'] = 0

work_df['reamining_contract'].fillna(0, inplace=True)
work_df.loc[work_df['reamining_contract'] > 0, 'reamining_contract'] = 1

for feature in ['download_avg', 'upload_avg']:
  work_df[feature].fillna(work_df[feature].mean(), inplace=True)

normalizations_dict = {}
for feature in ['subscription_age', 'bill_avg', 'download_avg', 'upload_avg']:
  work_df[f'{feature}_norm'] = (work_df[feature] - work_df[feature].min()) / (work_df[feature].max() - work_df[feature].min())
  normalizations_dict[feature] = {'min': work_df[feature].min(), 'max': work_df[feature].max()}

discrete_additional = ['service_failure_count_0', 'service_failure_count_1', 'service_failure_count_2', 'service_failure_count_3', 'service_failure_count_4', 'download_over_limit_0',
                       'download_over_limit_1', 'download_over_limit_2', 'download_over_limit_3', 'download_over_limit_4', 'download_over_limit_5', 'download_over_limit_6', 'download_over_limit_7',
                       'reamining_contract']
continuous_additial = ['subscription_age_norm', 'bill_avg_norm', 'download_avg_norm','upload_avg_norm']

# corr = work_df[discrete_initial + ['churn']].corr()
# sns.heatmap(corr, cmap="Blues", annot=True)

# corr = work_df[continuous_additial + ['churn']].corr()
# sns.heatmap(corr, cmap="Blues", annot=True)

# на основі аналізу кореляці з фін набору параметрів викидаємо upload_avg

final_df = work_df[['is_tv_subscriber', 'is_movie_package_subscriber', 'reamining_contract', 'service_failure_count_0', 'service_failure_count_1', 'service_failure_count_2', 'service_failure_count_3',
                    'service_failure_count_4', 'download_over_limit_0', 'download_over_limit_1', 'download_over_limit_2', 'download_over_limit_3', 'download_over_limit_4', 'download_over_limit_5', 
                    'download_over_limit_6', 'download_over_limit_7','subscription_age_norm', 'bill_avg_norm', 'download_avg_norm', 'churn']]

final_df.to_csv('final_df.csv')

X_train, X_test, y_train, y_test = train_test_split(final_df.loc[:, 'is_tv_subscriber' : 'download_avg_norm'], final_df['churn'], test_size = 0.3, random_state = 42, shuffle = True)

if __name__ == "__main__":
    print(X_train)