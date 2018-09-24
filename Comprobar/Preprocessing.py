#https://github.com/SITC-Twitter/Analisis_Sentimientos_Twitter

import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import re
import numpy as np

xml_data = "/Users/oscar/Downloads/Analisis_Sentimientos_Twitter-master/ficheros/Raw/general-tweets-train-tagged.xml"

def xml2df(xml_data):
    parser = ET.XMLParser(encoding='utf-8')
    tree = ET.parse(xml_data, parser)
    root = tree.getroot()
    all_records = []
    #headers = ['tweetid','user','content','date','lang','topic','polarity_value','polarity_type']
    headers = ['content','polarity_value']

    for tweet in root.iter('tweet'):
        record = []
        #record.append(tweet.find('tweetid').text)
        #record.append(tweet.find('user').text)
        record.append(tweet.find('content').text)
        #record.append(tweet.find('date').text)
        #record.append(tweet.find('lang').text)
        
        t = ""
        #for topics in tweet.iter('topics'):
        #    for topic in topics.iter('topic'):
        #        t = t+" "+topic.text
        #record.append(t)

        for sentiments in tweet.iter('sentiments'):
            #pol_ent = ""
            pol_val = ""
            #pol_type = ""
            for polarity in sentiments.iter('polarity'):
                #pol_ent = pol_ent+" "+polarity.find('entity').text
                pol_val = pol_val+" "+polarity.find('value').text
                #pol_type = pol_type+" "+polarity.find('type').text
            #record.append(pol_ent)
            record.append(pol_val)
            #record.append(pol_type)
                
        all_records.append(record)
    return pd.DataFrame(all_records, columns=headers)

tweets_df = xml2df(xml_data)

tweets_df = tweets_df[['content','polarity_value']]
for i in range(0,tweets_df.shape[0]):
    tweets_df.loc[i,'polarity_value'] = tweets_df.loc[i,'polarity_value'].split()[0] 
    #tweets_df.loc[i,'polarity_type'] = tweets_df.loc[i,'polarity_type'].split()[0]

car=['\n']

def borrar_caracteres(texto):
    for caracter in car:
        texto=texto.replace(caracter,"")
    return texto

emoji_pattern = re.compile(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+', 
    re.UNICODE)

for i in range(0,tweets_df.shape[0]):
    tweets_df.content[i]=borrar_caracteres(tweets_df.content[i])
    tweets_df.content[i]=tweets_df.content[i].strip()
    tweets_df.content[i]=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweets_df.content[i])
    tweets_df.content[i]=emoji_pattern.sub(r'', tweets_df.content[i])

tweets_df = tweets_df.query('polarity_value != "NONE"')
tweets_df = tweets_df.reset_index(drop=True)
tw_with_links = tweets_df.content.str.contains('http.*$')
tw_with_links.value_counts(normalize=False)
np.where(tw_with_links == True)
tweets_df[['content', 'polarity_value']].to_excel('/Users/oscar/Desktop/clean_tweets.xlsx', header=True, index=False)
print(tweets_df.head())
