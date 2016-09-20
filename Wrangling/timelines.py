import pandas as pd
import numpy as np

intakes = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/EarlyCleaning/in_cleaned.csv')
outcomes = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/EarlyCleaning/out_cleaned.csv')
df = pd.concat([intakes, outcomes])

#select relevant columns
cols = ['age','datetime','fixed','id','intake_condition','intake_type','outcome_type','outcome_detail','species']
df = df[cols]

df.sort_values(['id','datetime'])

#make sure these all have both income and outcome
counts = dict(df.id.value_counts())
solos = {}
odds = {}
for item in counts:
    if counts[item] == 1:
        solos[item] = counts[item]
    elif counts[item]%2 != 0:
        odds[item] = counts[item]
len(solos) #this makes up 2% of the data. It's not ideal but we need to delete it as it's not useful to only have one side of the story.
len(odds) #this only makes up 0.2% of the data. It's not worth trying to deal with.
for item in solos:
    df = df[df.id != item]
for item in odds:
    df = df[df.id != item]

#put these into events
events = df.sort_values(['id','datetime'])
events = events.reset_index().drop('index', axis=1)

timelines = pd.DataFrame()
for i in range(0,len(events),2):
    info = {
        'id': events.iloc[i].id,
        'species': events.iloc[i].species,
        'id_check': events.iloc[i+1].id,
        'intake_time' : events.iloc[i].datetime,
        'intake_type': events.iloc[i].intake_type,
        'age_in' : events.iloc[i].age,
        'condition_in' : events.iloc[i].intake_condition,
        'age_out' : events.iloc[i+1].age,
        'outcome_time' : events.iloc[i+1].datetime,
        'outcome_type': events.iloc[i+1].outcome_type,
        'outcome_detail': events.iloc[i+1].outcome_detail,
        'fixed_in': events.iloc[i].fixed,
        'fixed_out': events.iloc[i+1].fixed,
         }
    timelines = timelines.append(info, ignore_index=True)

timelines.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Wrangling/timelines.csv')

timelines = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Wrangling/timelines.csv')
timelines.isnull().sum()

#looks like ~110 got switched backwards (intake recorded before outcome)- need to fix
backwards = list(timelines[timelines.intake_type.isnull()].id)
mask = events['id'].isin(backwards)
backwardsdf = events.loc[mask]
backwardsdf.reset_index(inplace=True)
backwardsdf.drop('index', axis=1, inplace=True)
backwardsdf = backwardsdf.sort_values(['id','datetime'], ascending=False)
backwardsdf

backwardstimelines = pd.DataFrame()
for i in range(0,len(backwardsdf),2):
    info = {
        'id': backwardsdf.iloc[i].id,
        'species': events.iloc[i].species,
        'id_check': backwardsdf.iloc[i+1].id,
        'intake_time' : backwardsdf.iloc[i].datetime,
        'intake_type': backwardsdf.iloc[i].intake_type,
        'age_in' : backwardsdf.iloc[i].age,
        'condition_in' : backwardsdf.iloc[i].intake_condition,
        'age_out' : backwardsdf.iloc[i+1].age,
        'outcome_time' : backwardsdf.iloc[i+1].datetime,
        'outcome_type': backwardsdf.iloc[i+1].outcome_type,
        'outcome_detail': backwardsdf.iloc[i+1].outcome_detail,
        'fixed_in': backwardsdf.iloc[i].fixed,
        'fixed_out': backwardsdf.iloc[i+1].fixed,
         }
    backwardstimelines = backwardstimelines.append(info, ignore_index=True)

backwardstimelines

mask = timelines['id'].isin(backwards)
timelines = timelines.loc[~mask]
timelines = pd.concat([timelines, backwardstimelines])
len(timelines)
timelines.drop_duplicates(inplace=True)
timelines.dropna(subset=['intake_type','outcome_type'], inplace=True)
timelines.drop('Unnamed: 0', axis=1, inplace=True)

timelines['fixed_in_care'] = (timelines.fixed_out - timelines.fixed_in)
timelines['intake_time'] = pd.to_datetime(timelines['intake_time'])
timelines['outcome_time'] = pd.to_datetime(timelines['outcome_time'])
timelines['time_in_care'] = (timelines.outcome_time - timelines.intake_time)
timelines['time_in_care'] = timelines['time_in_care'].apply(lambda x: x.total_seconds())
timelines['time_in_care'] = timelines['time_in_care']/86400
timelines

timelines.dtypes
timelines.drop('id_check', axis=1, inplace=True)
timelines = timelines.sort_values(['id','intake_time'])
timelines.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Wrangling/timelines.csv')

dog_timelines = timelines[timelines.species == 'Dog']
dog_timelines.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dog_timelines.csv')
cat_timelines = timelines[timelines.species == 'Cat']
cat_timelines.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Cats/cat_timelines.csv')
other_timelines = timelines[timelines.species == 'Other']
other_timelines.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/OtherAnimals/other_timelines.csv')
