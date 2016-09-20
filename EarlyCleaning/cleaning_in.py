import pandas as pd
import re
import numpy as np

indf = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/EarlyCleaning/in_fixedcommas.csv')
indf = indf.drop_duplicates() #drops 6
indf.dtypes
indf.head()

#Make column names more useable
indf.columns
    #Index([u'Animal ID', u'Name', u'Intake Date', u'Intake Time', u'Found Location', u'Intake Type',
    #u'Intake Condition', u'Animal Type', u'Sex upon Intake', u'Age upon Intake', u'Breed', u'Color']
    #all have dtype 'object'
indf.rename(inplace=True, columns={'Animal ID':'id', 'Name':'name', 'Intake Date':'date', 'Intake Time':'time', \
    'Found Location':'found_loc', 'Intake Type':'intake_type', 'Intake Condition':'intake_condition', \
    'Animal Type':'species', 'Sex upon Intake':'sex', 'Age upon Intake':'age', 'Breed':'breed', 'Color':'color'})

#Check for null values
indf.isnull().sum()
#Name 11983, Sex upon Intake 1 (A667395), Age upon Intake 1 (A712959), Breed 23, Color 50
#Fill name with "None" since they're all unique, drop the single animals with missing intake data
#Handle breed and color later - there is likely an unknown class already
indf.dropna(subset=['sex','age'], inplace=True)
indf.name.fillna('None', inplace=True)

# Fix column by column:

#get rid of asterisks in names
indf['name'] = indf['name'].str.replace('*', '')
#check to make sure no more asterisks
for item in indf.name:
    x = str(item)
    if re.search(r'\*',x) != None:
        print indf.loc[indf.name == x]

#fix times
indf['datetime'] = indf.date+' '+indf.time
indf['datetime'] = pd.to_datetime(indf.datetime, errors='coerce')
indf.drop('date', axis=1, inplace=True)
indf.drop('time', axis=1, inplace=True)
indf.dropna(subset=['datetime'], inplace=True)

#determine if location is useful - potentially, will keep in dataframe but may not get to it
indf.found_loc.value_counts()

#organize intake information - type looks fairly clean, no need for changes
indf.intake_type.value_counts()
indf.intake_condition.value_counts()

#clean species information - bird and livestock are so small, group into Other
indf.species.replace('Bird','Other',inplace=True)
indf.species.replace('Livestock','Other',inplace=True)
indf.species.value_counts()

#sex column includes information on wheter animal is spayed/neutered and its sex
#establish spay/neuter info
def fixed(x):
    if (('Neutered' in x) or ('Spayed' in x)):
        fixed = 1
    elif ('Intact' in x):
        fixed = 0
    else:
        fixed = np.nan
    return fixed
indf['fixed'] = indf.sex.apply(fixed)

#establish animal's sex
def ismale(x):
    if ('Male' in x):
        male = 1
    elif ('Female' in x):
        male = 0
    else:
        male = np.nan
    return male
indf['male'] = indf.sex.apply(ismale)

indf.drop('sex', axis=1, inplace=True)

# ages are strings, need them as numbers
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

indf['age'] = indf.age.apply(agechange)

# will deal with breed and color in individual dataframes depending on importance

#export to csv
indf.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/EarlyCleaning/in_cleaned.csv')
