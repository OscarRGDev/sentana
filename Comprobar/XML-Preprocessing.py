import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import re
import numpy as np

data="/Users/oscar/Downloads/Analisis_Sentimientos_Twitter-master/ficheros/Raw/spain_geocode.xlsx"

def DataCreator(data):
    parser = ET.XMLParser(encoding='utf-8')
    tree = ET.parse(data, parser)
    root = tree.getroot()
    all_records = []
    headers = ['tweetid','user','content','date','lang','topic','polarity_value','polarity_type']

    for tweet in root.iter('tweet'):
        record = []
        record.append(tweet.find('tweetid').text)
        record.append(tweet.find('user').text)
        record.append(tweet.find('content').text)
        record.append(tweet.find('date').text)
        record.append(tweet.find('lang').text)
        
        t = ""
        for topics in tweet.iter('topics'):
            for topic in topics.iter('topic'):
                t = t+" "+topic.text
        record.append(t)

        for sentiments in tweet.iter('sentiments'):
            #pol_ent = ""
            pol_val = ""
            pol_type = ""
            for polarity in sentiments.iter('polarity'):
                #pol_ent = pol_ent+" "+polarity.find('entity').text
                pol_val = pol_val+" "+polarity.find('value').text
                pol_type = pol_type+" "+polarity.find('type').text
            #record.append(pol_ent)
            record.append(pol_val)
            record.append(pol_type)
                
        all_records.append(record)
    return pd.DataFrame(all_records, columns=headers)

tweets_df = DataCreator(data)

#Preprocessing:

tweets_df = tweets_df[['content','polarity_value']]

for tweet in range(0, tweets_df.shape[0]):
    tweets_df.loc[tweet,'polarity_value'] = tweets_df.loc[tweet, 'polarity_value'].split()[0]

for tweet in range(0, tweets_df.shape[0]):
    tweets_df.content[tweet] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweets_df.content[tweet])

tweets_df = tweets_df.query('polarity_value != "NONE"')
tweets_df = tweets_df.reset_index(drop=True)
tweets_df['polarity'] = 1 #Negative
tweets_df.polarity_bin[tweets_df.polarity_value.isin(['P', 'P+'])] = 2 #Positive
tweets_df.polarity_bin[tweets_df.polarity_value.isin(['NEU'])] = 3
#None = neutro??? ToDo!!!

tweets_df[['content','polarity']].to_excel('/User/oscar/Desktop/dataSet.xlsx', header=True, index=False)