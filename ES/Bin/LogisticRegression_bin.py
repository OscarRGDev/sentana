import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from mlxtend.preprocessing import DenseTransformer

from string import punctuation                                 # SVM
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB                       # MultinomialNB
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB                          # GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib
from sklearn.exceptions import ConvergenceWarning

import warnings
#Remove DeprecationWarning and ConvergenceWarning Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


#Train:
tweets_df = pd.read_excel('/Users/oscar/test.xlsx', header=0, encoding='iso8859_15')

#Tweets to predict:
tweets = pd.read_excel('/Users/oscar/Desktop/Sentiment/conneutros/Test_En.xlsx', header=0, encoding='iso8859_15')

#Stopwords + spanish "special" punctuation
spa_stop = stopwords.words('spanish')
#spa_stop = stopwords.words('spanish_Preprocesed1')
#spa_stop = stopwords.words('spanish_RAE')
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

#Defining X and y for training:
X = []
for i in range(tweets_df.shape[0]):
    X.append(tweets_df.iloc[i][0])
y = np.array(tweets_df['polarity_bin'])

#defining Train and test:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)

#Defining Model Pipeline for Logistic Regression:
model_Logistic_Regression = Pipeline([
    #Def stopword for model and Tf-idf  for the Tweets
    ('tfidf', TfidfVectorizer(stop_words=spa_stop)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag')))
])

#Defining Parameters:
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}


GridSearch_Log_Reg = GridSearchCV(model_Logistic_Regression, parameters, n_jobs=-1)
GridSearch_Log_Reg = GridSearch_Log_Reg.fit(X, y)
print('Training the Model please be patient.')
print('\n')
print('GridSearch Logistic Regression best score: \n',GridSearch_Log_Reg.best_score_)
print('\n')
print('GridSearch Logistic Regression best parameters: \n',GridSearch_Log_Reg.best_params_)

#Fit the model with Train
model_Logistic_Regression.fit(X_train, y_train)

pred = model_Logistic_Regression.predict(X_test)
#Defining Confusion Matrix
confusion_matrix(pred, y_test)
print('Fit X, y with Logistic Regression Algorithm ',model_Logistic_Regression.fit(X, y))
#model_Logistic_Regression.fit(X, y)
#Export pkl
joblib.dump(GridSearch_Log_Reg, "model_LogistikRegression.pkl")
#import pkl
model_Logistic_Regressionodel = joblib.load("model_LogistikRegression.pkl")

y_preds = model_Logistic_Regression.predict(X_test)
print('\n')
print('\n')
print('accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
print(classification_report(y_test, y_preds))

#Defining Range prediction from Sheet. 
tweet_preds = model_Logistic_Regression.predict(tweets['content'])
#Defining Range to save the prediction
df_tweet_preds = tweets.copy()
df_tweet_preds['Polarity'] = tweet_preds
df_tweet_preds.shape

index = random.sample(range(tweet_preds.shape[0]), 20)
for text, sentiment in zip(df_tweet_preds.content[index],
                           df_tweet_preds.Polarity[index]):                 #Para SVM escribir en esta linea df_tweet_preds.Polarity[index]):
    print (sentiment, '===> ', text, '\n')