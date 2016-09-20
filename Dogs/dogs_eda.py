import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

indivs = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_info.csv')
indivs.drop('Unnamed: 0', axis=1, inplace=True)

timelines = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dog_timelines.csv')
timelines.drop('Unnamed: 0', axis=1, inplace=True)

df = timelines.merge(indivs, how='left', on='id')
df.columns
len(df)
df.isnull().sum()
#somehow we don't have data for about 300 dogs demographics. This makes up 1.4% of our data, so may be worth figuring out later, but for now we will drop it.
# df.dropna(subset=['color'], inplace=True)
#we also don't have sex/fixed info for 88 dogs, or 0.4% of the data. We will drop these individuals as well.
df.dropna(subset=['male'], inplace=True)

#a few animals (26) got their dates mixed up and have negative time in care. this could be an issue so let's drop those.
df = df[df.time_in_care >= 0]

df.describe()
# categorical = ['condition_in', 'fixed_in', 'fixed_out', 'intake_type', 'outcome_detail', 'outcome_type', 'fixed_in_care', 'breed', 'male']
# continuous = ['age_in', 'age_out', 'time_in_care']
# other = ['intake_time', 'outcome_time']

# for item in categorical:
#     print item
#     print df[item].value_counts()/len(df)
#     print

#outcome type: ignore died, missing, disposal (each <1% of data)

#explore condition in
#vast majority (90%) is "normal" - can go with normal and other
normal = df[df['condition_in'] == 'Normal']
abnormal = df[df['condition_in'] != 'Normal']

norm = pd.DataFrame(normal.outcome_type.value_counts(normalize=True))
abnorm = pd.DataFrame(abnormal.outcome_type.value_counts(normalize=True))

outcome_by_intake_condition = norm.join(abnorm, how='inner', lsuffix='_normal_intake', rsuffix='_not_normal_intake')
#normal animals much more likely to be adopted or returned to owner
#abnormal animals much more likely to be transfered or euthanised

normal.groupby('outcome_type').intake_type.value_counts(normalize=True)
abnormal.groupby('outcome_type').intake_type.value_counts(normalize=True)
#comparison: ratios very similar within outcome type classes between normal and abnormal.

#explore intake type
df.intake_type.value_counts(normalize=True)
df.groupby('outcome_type').intake_type.value_counts(normalize=True) #less useful, matches overall portions by intake_type

for x in df.intake_type.value_counts().index:
    name = '/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/' + str(x) + ".png"
    plot = df[df.intake_type == x].outcome_type.value_counts().plot.pie(figsize=(6,6), title=x)
    fig = plot.get_figure()
    fig.savefig(name)
    plt.close(fig)

df.groupby('intake_type').outcome_type.value_counts(normalize=True).plot.pie()
#euthanasia requests are most likely to be euthanised (87% of the time)

#owner surrenders are most likely adopted (52%) or Transfered (32%), though about 9% euthanised and 7% returned to owner
#return to owner average is 7 days (std of 12 days) while adoption average is 22 days (std 38) and overall is 13 days (std 28)
#thought this could be due to health needs, but of the animals surrendered and then returned, 92% were "normal" upon intake and 86% did not receive neutering in care
#owner surrenders take almost exactly the same amount of time to be adopted as overall adoptions (22 days, std 38 days (std 39 for overall))
#owner surrenders are euthanised after about the same amount of time as overall dataset as well (8.6 days, std 16 vs 7.3 std 15)

#public assists get returned to owner most frequently (74% of the time), though it takes 7 days (std 4) to accomplish this
#otherwise, they get adopted (12%), transferred (9%), or, rarely, euthanised (6%, about the same as overall dataset )

#strays, interestingly, are euthanised the least of any intake type (only 4%) and stay on average 10 days (longer than the 8.5 overall)
#they get adopted at 44%, second only to owner surrenders, though it does take much longer (22 days, std 36 days)
#they also get transferred to partners frequently, which are usually rescue groups. 22% meet this fate, after waiting an average of 12 days

'''
useful temp code for this section

df.describe()
df[df['outcome_type']=='Euthanasia'].describe()
temp = df[df['intake_type'] == 'Stray']
temp[temp['outcome_type']=='Transfer'].describe()
'''

#does being fixed at either end or getting fixed in care matter
fixed = df[['fixed_in', 'fixed_out', 'fixed_in_care', 'outcome_type', 'time_in_care']]
for item in fixed:
    print item
    print fixed[item].value_counts(normalize=True)
fixed.groupby('fixed_in').outcome_type.value_counts(normalize=True)

fixed_at_intake = df[df['fixed_in'] == 1]
fixed_at_outcome = df[df['fixed_out'] == 1]
fixed_in_care = df[df['fixed_in_care'] == 1]
not_fixed_at_intake = df[df['fixed_in'] == 0]
not_fixed_at_outcome = df[df['fixed_out'] == 0]
not_fixed_in_care = df[df['fixed_in_care'] == 0]

fixed_at_intake = pd.DataFrame(fixed_at_intake.outcome_type.value_counts(normalize=True))
fixed_at_outcome = pd.DataFrame(fixed_at_outcome.outcome_type.value_counts(normalize=True))
fixed_in_care = pd.DataFrame(fixed_in_care.outcome_type.value_counts(normalize=True))
not_fixed_at_intake = pd.DataFrame(not_fixed_at_intake.outcome_type.value_counts(normalize=True))
not_fixed_at_outcome = pd.DataFrame(not_fixed_at_outcome.outcome_type.value_counts(normalize=True))
not_fixed_in_care = pd.DataFrame(not_fixed_in_care.outcome_type.value_counts(normalize=True))

outcome_by_fixing = pd.concat([fixed_at_intake,not_fixed_at_intake,fixed_at_outcome,not_fixed_at_outcome,fixed_in_care,not_fixed_in_care], axis=1, keys=['intake','not_intake','outcome','not_outcome','in_care','not_in_care'])
outcome_by_fixing
#Died, disposal, and missing are so small we're going to ignore them here
#In most groups, adoption is the most likely outcome, as withe the population in general (overall average is 43%)
#The main exception to this is that animals that are already fixed at intake are more likely to be returned to owner(47%)
#69% of animals that are fixed in care get adopted. This may be due to Austin only fixing animals once they have a home.
#animals that are returned to owner or transferred are also fixed while in care.
#animals that are not fixed at income and not fixed in care are overwhelmingly transfered to partners (53%) or returned to owners (26%)
#while being fixed or not at intake doesn't increase the likelihood for euthanasia, there is a correlation between euthanasia and
    #not getting fixed - 10% of animals that don't receive fixing in care, and 15% of animals that are not ever fixed get euthanised.

#investigate effect of sex
sexinfo = df[['male','outcome_type','outcome_detail']]
sexinfo.male.value_counts()/len(sexinfo)
sexinfo.groupby('outcome_type').male.value_counts(normalize=True)
#died is even split, disposal and missing are too small to care
#53% of the dogs are male. Adoptions for males are only 51%, small disadvantage. 57% of euthanised dogs are male. Same with dogs returned to owner.
sexinfo[sexinfo.outcome_type == 'Euthanasia'].groupby('male').outcome_detail.value_counts(normalize=True)
#for those euthanised - females had 49% suffering, 33% aggressive, 9% behavior, 8% health. Males had 37% Suffering, 39% aggressive, 12% behavior, 11% health.
#transfer proportions are in line with overall proportions
#males seem to have a small disadvantage

#look at detail for euthanasia only
sad = df[['outcome_type', 'outcome_detail', 'intake_type', 'condition_in']]
sad = sad[sad.outcome_type == 'Euthanasia']
sad.groupby(['outcome_detail']).intake_type.value_counts(normalize=True)
#all had justification. 52% health (suffering/rabies/medical), 47% action (aggressive/behavior)
#most rabies risks were owner surrenders, most behavior, medical and suffering were strays, aggressive were a more even mix
#behavior types were only documented for those euthanised, so it's very hard to tell proportion of "difficult" animals euthanized.

#explore age
adopted = df[df.outcome_type=='Adoption']
plt.scatter(adopted.age_in, adopted.time_in_care)
#hard to tell - will learn more in modeling phase

#simplify breed - check out whether mix has an effect, and whether certain groups are more likely to get adopted - and after how long


#investigate whether color matters at all

#explore time in care
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dummydf = df.copy()
dummydf['outcome_dummy'] = le.fit_transform(dummydf.outcome_type)
le.classes_
dummydf.groupby('outcome_dummy').time_in_care.plot(kind='hist')


for item in continuous:
    print item
    plt.hist(df[item])
    plt.show()

#plot time in care
plt.hist(timelines.time_in_care, bins=50)
plt.axis([-800,25,0,3200])
plt.show()

plt.plot(timelines.outcome_type)
relevant = []
df[relevant].to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_tomodel.csv')
