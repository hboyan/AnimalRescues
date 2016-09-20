import re

#note - need to go into dataframes and find commas mid address
fileout = open('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/in_parsed.csv','w')

def fix_commas(myfile):
    output = []
    text = myfile.read().split('\n')
    for i in range(len(text)):
        y = re.split(r'([AP]M,)([\s\S]*\(TX\))',text[i])
        try:
            z = y[0] +y[1] + y[2].replace(',','') + y[3]
        except:
            z = text[i]
        output.append(z+'\n')
    return output

output1 = fix_commas(open('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/Austin_Animal_Center_FY14_Intakes.csv','r'))
output2 = fix_commas(open('/Users/boyan/Dropbox/DCIWork/Capstone/Data/AustinTake2/Austin_Animal_Center_FY15_Intakes__Updated_Hourly_.csv'))
len(output1) + len(output2)

for item in output1:
    fileout.write(item)
for item in output2:
    fileout.write(item)
