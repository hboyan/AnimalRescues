import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#repeat the important setup points from eda
indivs = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_info.csv')
indivs.drop('Unnamed: 0', axis=1, inplace=True)
timelines = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dog_timelines.csv')
timelines.drop('Unnamed: 0', axis=1, inplace=True)
df = timelines.merge(indivs, how='left', on='id')
df.dropna(subset=['colors'], inplace=True)
df.dropna(subset=['male'], inplace=True)
df = df[df.time_in_care >= 0]
df.drop('species_x', inplace=True, axis=1)
df.drop('species_y', inplace=True, axis=1)
df.drop('name', inplace=True, axis=1)
df.drop('breeds', inplace=True, axis=1)
df.drop('groups', inplace=True, axis=1)
df.drop('colors', inplace=True, axis=1)

df.intake_time = df.intake_time.apply(lambda x: pd.to_datetime(x))
df.outcome_time = df.outcome_time.apply(lambda x: pd.to_datetime(x))
df.mix = df.mix.apply(lambda x: float(x))

df.dtypes
condition_in_dummies = pd.get_dummies(df.condition_in, prefix='intake_condition')
outcome_detail_dummies = pd.get_dummies(df.outcome_detail, prefix='outcome_detail')
outcome_type_dummies = pd.get_dummies(df.outcome_type, prefix='outcome_type')
intake_type_dummies = pd.get_dummies(df.intake_type, prefix='intake_type')

df_done = pd.concat([df,condition_in_dummies,outcome_detail_dummies,outcome_type_dummies,intake_type_dummies], axis=1)
df_done.to_csv('dogs_tomodel.csv')
