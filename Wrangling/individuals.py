import pandas as pd
import numpy as np

intakes = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/EarlyCleaning/in_cleaned.csv')
outcomes = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/EarlyCleaning/out_cleaned.csv')
df = pd.concat([intakes, outcomes])

#pull out what's useful for describing the individual (not the event)
df = df[['breed','color','id','male','species','name']]
df.drop_duplicates(inplace=True)
df.dropna(subset=['breed','color'], inplace=True)

df.isnull().sum()

#check for duplicates (animals with mismatched data)
(df.id.value_counts() > 1).sum() #165
(df.id.value_counts() == 1).sum() #35495
#only 0.5% of the data is this messy/unmatched set. I will drop them to avoid confusion.

#would deleting null males fix any of the duplicates?
nosex = list(df[df.male.isnull()].id)
mask = df['id'].isin(nosex)
df.loc[mask].male.value_counts()
#no, only one animal shows as both male and nan

vals = dict(df.id.value_counts())
for item in vals:
    if vals[item] > 1:
        df = df[df.id != item]

df

#breed will require different manipulation for each species, so will do in their individual cleaning

'''
#should we format color here?
#get a list of all the colors shown
allcolors = {}
for item in df.color:
    colors = item.replace("'",'').replace('[','').replace(']','').replace('/',', ').split(', ')
    for item in colors:
        if item not in allcolors:
            allcolors[item] = 1
        else:
            allcolors[item] = allcolors[item] + 1
    colorsdf = pd.DataFrame([allcolors]).transpose()

colorsdf.sort_values(0, ascending=False)
colorsdf[0].sum()
#I've decided against doing anything with colors, as this will again probably be more relevant in the individual data frames
'''
cats = df[df.species == 'Cat']
dogs = df[df.species == 'Dog']
others = df[df.species == 'Other']

# df.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Wrangling/individuals.csv')
# cats.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Cats/cats_info.csv')
dogs.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_info.csv')
# others.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/OtherAnimals/others_info.csv')
