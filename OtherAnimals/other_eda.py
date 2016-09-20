import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

indivs = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/OtherAnimals/others_info.csv')
timelines = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/OtherAnimals/other_timelines.csv')
timelines.drop('Unnamed: 0', axis=1, inplace=True)
timelines.columns
indivs.columns

df = timelines.merge(indivs, how='left', on='id')
len(df)
df.isnull().sum()

categorical = ['condition_in', 'fixed_in', 'fixed_out', 'intake_type', 'outcome_detail', 'outcome_type', 'fixed_in_care', 'breed', 'color']
continuous = ['age_in', 'age_out', 'time_in_care']
other = ['intake_time', 'outcome_time']

for item in categorical:
    print item
    print df[item].value_counts()/len(df)
    print

df.breed = df.breed.apply(lambda x: str(x).replace(' Mix', ''))

for item in continuous:
    print item
    plt.hist(df[item])
    plt.show()

plt.hist(timelines.age_in, bins=200)
plt.axis([0,5,0,500])
plt.show()

plt.hist(timelines.age_out, bins=200)
plt.axis([0,5,0,500])
plt.show()

plt.hist(timelines.time_in_care, bins=200)
plt.axis([0,5,0,500])
plt.show()

#they don't stay in long, and they're young when they get there
#fixed and sex information is not meaningful - only present in 13% of the data
#we're left with condition, intake type, breed, color
#home intakes (adoptions (randoms - ferrets, chickens), transfers (mostly rodents), returns to owner (randoms - tortoise, ferret)) are pets - strays, surrenders, public assists (rescues)
#deaths (died (41 bats, 6 raccoons), disposal (112 bats, 14 raccoons), euthanasia (846 Bats, 382 Raccoons, then opossums/skunks/squirrels/foxes)) are overwhelmingly wildlife
#most euthanasia is rabies risk (1078) or suffering (282)

df[df.outcome_type == 'Died'].outcome_detail.value_counts()

df.groupby(['outcome_type']).intake_type.value_counts()

#use decision trees here, because there are such large chunks making these choices
relevant = ['condition_in', 'intake_type', 'outcome_detail', 'outcome_type', 'breed']
df[relevant].to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/OtherAnimals/other_tomodel.csv')
