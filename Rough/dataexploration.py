import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

animals = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/animals.csv')
timelines = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/timelines.csv')

df = timelines.merge(animals, how='left', on='id')
df = df.drop('Unnamed: 0_x', axis=1)
df = df.drop('Unnamed: 0_y', axis=1)
df = df.drop('id_check', axis=1)
df.dtypes
df['intake_time'] = pd.to_datetime(df['intake_time'])
df['outcome_time'] = pd.to_datetime(df['outcome_time'])
df['fixed_in_care'] = (df.fixed_in != df.fixed_out).astype('float')

df['time_held'] = df.outcome_time - df.intake_time
df[df['fixed_in_care']==False].outcome_type.value_counts()
df[df['fixed_in_care']==True].outcome_type.value_counts()

%matplotlib inline
stay_times = df.time_held.astype('timedelta64[D]')
stay_times

turnarounds0 = stay_times[stay_times==0]
len(turnarounds0)
turnarounds1 = stay_times[stay_times==1]
len(turnarounds1)

plt.axis([0,5,0,10000])
plt.hist(stay_times[stay_times<=5], bins=6)
plt.show()

plt.axis([0,100,0,10000])
plt.hist(stay_times[stay_times>5], bins=50)
plt.show()

attribs = list(df.columns)
attribs.remove('outcome_type')
attribs.remove('id')
attribs


df.outcome_type.value_counts()

df.condition_in.value_counts()
df[df.outcome_type == 'Missing'].outcome_detail.value_counts()

df

mappydf = df[df.male.notnull()]
mappydf = mappydf[mappydf.intake_type.notnull()]
mappydf = mappydf[mappydf.outcome_type.notnull()]
mappydf = mappydf[mappydf.fixed_in.notnull()]

mappydf.isnull().sum()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(mappydf.condition_in)
mappydf.condition_in = le.fit_transform(mappydf['condition_in'])

le.fit(mappydf.intake_type)
mappydf.intake_type = le.fit_transform(mappydf['intake_type'])

le.fit(mappydf.outcome_detail)
mappydf.outcome_detail = le.fit_transform(mappydf['outcome_detail'])

le.fit(mappydf.outcome_type)
mappydf.outcome_type = le.fit_transform(mappydf['outcome_type'])

le.fit(mappydf.species)
mappydf.species = le.fit_transform(mappydf['species'])

mappydf['male'] = mappydf['male'].astype('int')
mappydf

mappydf.drop('name', axis=1, inplace=True)
mappydf.drop('color', axis=1, inplace=True)
mappydf.drop('breed', axis=1, inplace=True)

mappydf.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/mappydf.csv')
