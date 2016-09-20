import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dogs = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/dogs.csv')
timelines = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/timelines.csv')
df = timelines.merge(dogs, how='left', on='id')

df.columns
df.drop(['Unnamed: 0_x', 'Unnamed: 0_y', 'id_check', 'species', 'name'], axis=1, inplace=True)
df = df.dropna()
df['fixed_in_care'] = (df['fixed_in'] != df['fixed_out']).astype('int')
df['male'] = df['male'].astype('int')
df['mix'] = df['mix'].astype('int')
df['fixed_in_care'] = (df['fixed_in'] != df['fixed_out']).astype('int')
df['outcome_time'] = df['outcome_time'].apply(lambda x: pd.to_datetime(x))
df['intake_time'] = df['intake_time'].apply(lambda x: pd.to_datetime(x))
df['time_in_care'] = df['outcome_time'] - df['intake_time']
df['time_in_care'] = df['time_in_care'].apply(lambda x: x.total_seconds())
df = df.dropna()

# def label(col):
#     output = col + '_le'
#     reference = output + 'ref'
#     le = LabelEncoder()
#     le.fit(df[col])
#     df[output] = le.transform(df[col])
#     return pd.DataFrame(le.classes_)
#
# condition_in_ref = label('condition_in')
# outcome_detail_ref = label('outcome_detail')
# outcome_type_ref = label('outcome_type')

breedguide = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/breeds.csv')
grouplist = breedguide['American Kennel Club'].value_counts().index

allcolors = []
for item in df.colors.value_counts().index:
    colors = item.replace("'",'').replace('[','').replace(']','').replace('/',', ').split(', ')
    for item in colors:
        if item not in allcolors:
            allcolors.append(item)

def myDummies(df, col, xlist):
    for item in xlist:
        df[item] = df[col].apply(lambda x: item in x)
        df[item] = df[item].astype('int')

myDummies(df, 'groups', grouplist)
myDummies(df, 'colors', allcolors)

df = pd.concat([df, pd.get_dummies(df['condition_in'], prefix='cond_in')], axis=1)
df = pd.concat([df, pd.get_dummies(df['intake_type'], prefix='intake_type')], axis=1)

df = df[df.outcome_type != 'Disposal']
df = df[df.outcome_type != 'Missing']
df = df[df.outcome_type != 'Died']
df

X=df[cols]
X = X.set_index('id')

# targets = pd.DataFrame(df[['id', 'outcome_type', 'outcome_detail']])
# targets = pd.concat([targets, pd.get_dummies(df['outcome_type'], prefix='outcome_type')], axis=1)
# targets = pd.concat([targets, pd.get_dummies(df['outcome_detail'], prefix='outcome_detail')], axis=1)

# set id
not_useful = ['condition_in', 'intake_type', 'outcome_detail', 'outcome_type', 'breeds', 'colors', 'groups', 'intake_time', 'outcome_time']
cols = list(df.columns)
for item in not_useful:
    cols.remove(item)
cols

# targets
# targets.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/dog_dummies_targets.csv')
# df.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/dog_dummies_feats.csv')
