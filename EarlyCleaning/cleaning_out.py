import pandas as pd
import re
import numpy as np

out14 = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/RawData/Austin_Animal_Center_FY14_Outcomes.csv')
out15 = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/RawData/Austin_Animal_Center_FY15_Outcomes__Updated_Hourly_.csv')
outdf = pd.concat([out15, out14])

outdf = outdf.drop_duplicates() #drops 6

#rename columns - all have type object
outdf.columns
outdf.rename(inplace=True, columns={'Animal ID':'id', 'Name':'name', 'Outcome Date':'date', 'Outcome Time':'time', \
    'Outcome Type':'outcome_type', 'Outcome Subtype':'outcome_detail', \
    'Animal Type':'species', 'Sex upon Outcome':'sex', 'Age upon Outcome':'age', 'Breed':'breed', 'Color':'color'})

#check for and fix null values
outdf.isnull().sum()
#drop the single individual missing sex, and the 8 individuals missing outcome_type (since this is the target)
outdf.dropna(inplace=True, subset=['sex','outcome_type'])
#fill the empty names and empty outcome_details with "None"
outdf.name.fillna('None', inplace=True)
outdf.outcome_detail.fillna('None', inplace=True)

#column by column
#get rid of asterisks in names
outdf['name'] = outdf['name'].str.replace('*', '')
#check to make sure no more asterisks
for item in outdf.name:
    x = str(item)
    if re.search(r'\*',x) != None:
        print outdf.loc[outdf.name == x]

#fix times
outdf['datetime'] = outdf.date+' '+outdf.time
outdf['datetime'] = pd.to_datetime(outdf.datetime, errors='coerce')
outdf.drop('date', axis=1, inplace=True)
outdf.drop('time', axis=1, inplace=True)

#check outcome info
outdf.groupby('outcome_detail').outcome_type.value_counts()
#all potentially useful, some are small numbers and can get dropped in modeling

#clean species column
outdf.species.replace('Bird','Other',inplace=True)
outdf.species.replace('Livestock','Other',inplace=True)
outdf.species.value_counts()

#establish spay/neuter info
def fixed(x):
    if (('Neutered' in x) or ('Spayed' in x)):
        fixed = 1
    elif ('Intact' in x):
        fixed = 0
    else:
        fixed = np.nan
    return fixed
outdf['fixed'] = outdf.sex.apply(fixed)

#establish animal's sex
def ismale(x):
    if ('Male' in x):
        male = 1
    elif ('Female' in x):
        male = 0
    else:
        male = np.nan
    return male
outdf['male'] = outdf.sex.apply(ismale)

outdf.drop('sex', axis=1, inplace=True)

#make ages the right format
def agechange(age):
    newage = age.split()
    try:
        newage[0] = int(newage[0])
    except ValueError:
        return np.nan

    if newage[1] in ['year','years']:
        newage[1] = 365
    elif newage[1] in ['month','months']:
        newage[1] = 30
    elif newage[1] in ['week','weeks']:
        newage[1] = 7
    elif newage[1] in ['day','days']:
        newage[1] = 1

    newage = newage[0]*newage[1]
    return newage/365.0

outdf['age'] = outdf.age.apply(agechange)

#again, handle breed and color in modeling phase

outdf.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/EarlyCleaning/out_cleaned.csv')
