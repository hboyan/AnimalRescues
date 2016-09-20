import pandas as pd
import numpy as np

intakes = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/intakes.csv')
outcomes = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/outcomes.csv')
df = pd.concat([intakes, outcomes])
df = df.drop('Unnamed: 0', axis=1)
len(df)

df.sort_values(['id','datetime'])

attributes = ['breed','color','male','name','species']
animals = df[attributes+['id']]
animals = animals.drop_duplicates()
animals = animals[animals.id != 'A669371']

# #is this worth doing?
odds = []
vals = dict(animals.id.value_counts())
for item in vals:
    if vals[item] > 1:
        odds.append(item)
# len(odds) #213
#
# rows = 0
# for item in odds:
#     thisdf = df[df.id == item]
#     rows += len(thisdf)
# rows
# #799
# #yeah, i wanna know

def whatsdiff(df):
    diffs = []
    for col in df:
        if df[col].nunique() > 1:
            diffs.append(col)
    return diffs

diffs = {}
for item in odds:
    thisdf = animals[animals.id == item]
    diffs[item] = whatsdiff(thisdf)

diffs

def cleanup(mydict, mydf):
    # mydf['breed'] = mydf['breed'].str.replace('/', ' Mix ')
    mydf['color'] = mydf['color'].str.replace('/', ' ')

    for item in mydict:
        if 'name' in mydict[item]:
            oldname = list(mydf.loc[mydf.id == item].name)
            if 'None' in oldname:
                oldname.remove('None')
            newname = str(oldname).strip('[').strip(']').strip('\'').replace("', '", '/')
            mydf.ix[mydf.id == item, 'name'] = newname

        if 'breed' in mydict[item]:
            oldbreed = list(mydf.loc[mydf.id == item].breed)
            if 'None' in oldbreed:
                oldbreed.remove('None')
            if 'unknown' in oldbreed:
                oldbreed.remove('unknown')
            newbreed = str(oldbreed)
            mydf.ix[mydf.id == item, 'breed'] = str(newbreed)

        if 'color' in mydict[item]:
            oldcolor = list(mydf.loc[mydf.id == item].color.str.split('//'))
            if 'None' in oldcolor:
                oldname.remove('None')
            if 'unknown' in oldcolor:
                oldcolor.remove('unknown')
            newcolor = str(oldcolor).replace('[','').replace(']','').replace('\'',' ').replace("', '", '/').replace('/', ' ').replace(',','')
            finalcolor = []
            for word in newcolor.split(' '):
                if word not in finalcolor and len(word)>0:
                    finalcolor.append(word)
            mydf.ix[mydf.id == item, 'color'] = str(finalcolor)

cleanup(diffs, animals)

#check breedfix and colorfix
animals.loc[animals.id == 'A601696']

#check namefix
animals.loc[animals.id == 'A468026']

animals = animals.drop_duplicates()
animals
animals.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/animals.csv')
