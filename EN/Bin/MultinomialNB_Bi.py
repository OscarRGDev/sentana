from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB                          # GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from mlxtend.preprocessing import DenseTransformer
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB                       # MultinomialNB
from sklearn.svm import SVC                                         # SVM
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
import warnings
#Remove DeprecationWarning and ConvergenceWarning Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


####importar archivos:
#tweets_df = pd.read_excel('Desktop/bla_4.xlsx' , header=0, encoding='iso8859_15')
#tweets = pd.read_excel('Downloads/Analisis_Sentimientos_Twitter-master/ficheros/TweetsConTopic/tweets_spainGeo_topic_nadia.xlsx', header=0, encoding='iso8859_15')


#Train:
tweets_df = pd.read_excel('/Users/oscar/test.xlsx', header=0, encoding='iso8859_15')

#Tweets to predict:
tweets = pd.read_excel('/Users/oscar/Desktop/Sentiment/conneutros/Test_En.xlsx', header=0, encoding='iso8859_15')

X_train, X_test, y_train, y_test = train_test_split(tweets_df['content'][:5000], tweets_df['polarity_bin'][:5000], test_size=0.2)

#Stopwords + spanish "special" punctuation
spa_stop = stopwords.words('spanish')
punctuation = list(punctuation)
punctuation.extend(['¿','!'])
spa_stop.extend(punctuation)
spa_stop.extend(['¿','!'])


#spanisch stemmer:
stemmer = SnowballStemmer('spanish')
#reduce_len=True: "waaaaayyyy" -> "waaayyy"
tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

def token_stemmer(token, stemmer):
    stemmed = []
    for i in token:
        stemmed.append(stemmer.stem(i))
    return stemmed

def tokenization(text):
    #remove all non-words:
    text = ''.join([non_w for non_w in text if non_w not in punctuation])
    
    token = tokenizer.tokenize(text)
    #Stemming:
    try:
        stem = token_stemmer(token,stemmer)
    except Exception as e:
        print(e)
        print(text)
        stem=['']
    return stem

pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words=spa_stop,
                            tokenizer = tokenization,
                            analyzer = 'word',
                            lowercase=True)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }


print('Training the Model please be patient.')
grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
grid.fit(X_train,y_train)

print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))          #Usar como creador del segundo modelo para recibir mejores resultados! https://towardsdatascience.com/a-production-ready-multi-class-text-classifier-96490408757

joblib.dump(grid, "twitter_sentiment.pkl")
model_NB = joblib.load("twitter_sentiment.pkl" )

y_preds = model_NB.predict(X_test)

print('accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
print(classification_report(y_test, y_preds))

tweet_preds = model_NB.predict(tweets['content'])

df_tweet_preds = tweets.copy()
df_tweet_preds['Polarity'] = tweet_preds
df_tweet_preds.shape

index = random.sample(range(tweet_preds.shape[0]), 20)
for text, sentiment in zip(df_tweet_preds.content[index],
                           df_tweet_preds.Polarity[index]):
    print (sentiment, '--', text, '\n')
