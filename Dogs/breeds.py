import pandas as pd
import warnings
warnings.filterwarnings("ignore")

breeds = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/RawData/breeds.csv')
breeds = breeds[['Breed', 'American Kennel Club']]
breeds.rename(inplace=True, columns={'American Kennel Club':'akc'})

dogs = pd.read_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_info.csv')
dogs.drop('Unnamed: 0',axis=1,inplace=True)
dogs.rename(columns={'breeds':'breed', 'colors':'color'}, inplace=True)
dogs

def to_list(df,inputcol,outputcol,splitchar):
    df[inputcol] = df[inputcol].str.replace(" Mix", '')
    df[inputcol] = df[inputcol].str.replace("\[\'", '')
    df[inputcol] = df[inputcol].str.replace("\'\]", '')
    df[inputcol] = df[inputcol].str.replace("\', \'", '/')
    df[outputcol] = df[inputcol].str.split(splitchar)
    df.drop(inputcol, axis=1, inplace=True)

to_list(dogs,'breed','breeds','/')
dogs['mix'] = dogs.breeds.apply(lambda x: len(x) >= 2)
dogs

allbreeds = dogs.breeds
breeds_present = []
for item in allbreeds:
    for breed in item:
        if breed not in breeds_present:
            breeds_present.append(breed)
pd.DataFrame(breeds_present).to_csv('breedlist.csv')
breeds_present
allbreeds

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,10), min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.fit_transform(breeds.Breed)
feature_names = tf.get_feature_names()
dense = tfidf_matrix.todense()
relationships = pd.DataFrame(dense, index=[breeds.Breed], columns=[feature_names])

group_dict = {}
for item in breeds_present:
    try:
        guess = relationships[item.lower()].sort_values(ascending=False).index[0]
    except:
        try:
            bestguess = 0
            bestscore = 0
            for word in item.split():
                try:
                    score = relationships[word.lower()].sort_values(ascending=False)[0]
                    if score > bestscore:
                        bestguess = relationships[word.lower()].sort_values(ascending=False).index[0]
                        bestscore = score
                    guess = bestguess
                except:
                    pass
        except:
            print 'error'
    group_dict[item] = guess

breeds

for key in group_dict:
    br = group_dict[key]
    x = breeds.loc[breeds.Breed == br]
    gr = list(x['akc'])[0]
    group_dict[key] = [br, gr]
group_dict

def groupmatch(x):
    grps = []
    for item in x:
        group = group_dict[item][1]
        if group not in grps and group != np.nan:
            grps.append(group)
    result = str(sorted(grps))
    result = result.replace(', nan', '')
    result = result.replace('nan, ', '')
    return result
dogs['groups'] = dogs.breeds.apply(groupmatch)
dogs[['breeds','groups']]

grouplist = list(breeds.akc.value_counts().index)
grouplist

for item in grouplist:
    dogs[item] = dogs.groups.apply(lambda x: int(item in str(x)))

# allcolors = dogs.color
# colors_present = []
# for item in allcolors:
#     colorlist = str(item).split('/')
#     print colorlist
#     for color in colorlist:
#         if color not in colors_present:
#             colors_present.append(color)
# colors_present
# to_list(dogs,'color','colors','/')
#
# for item in colors_present:
#     dogs[item] = dogs.colors.apply(lambda x: int(item in str(x)))

for col in dogs:
    if dogs[col].dtype == 'int64' and dogs[col].sum() < 100:
        dogs.drop(col, axis=1, inplace=True)
dogs.sum()

dogs.to_csv('/Users/boyan/Dropbox/DCIWork/Capstone/Dogs/dogs_info.csv')
