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
from sklearn.svm import LinearSVC

import warnings
#Remove DeprecationWarning and ConvergenceWarning Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


#Train:
tweets_df = pd.read_excel('/Users/oscar/Desktop/Sentiment/conneutros/Training_En.xlsx', header=0, encoding='iso8859_15')

#Tweets to predict:
tweets = pd.read_excel('/Users/oscar/Desktop/Sentiment/conneutros/Test_En.xlsx', header=0, encoding='iso8859_15')

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

#Defining X and y for training:
X = []
for i in range(tweets_df.shape[0]):
    X.append(tweets_df.iloc[i][0])
y = np.array(tweets_df['Polarity'])

#defining Train and test:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)

#Defining Model Pipeline for SVM:
model_SVM = Pipeline([
    ('vectorizer', CountVectorizer(strip_accents='ascii',
                            stop_words=spa_stop,
                            tokenizer = tokenization,
                            analyzer = 'word',
                            lowercase=True)), 
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))
 ])


#Defining Parameters:
parameters = {
    'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
    'tfidf__use_idf': (True, False)
}

GridSearch_SVM = GridSearchCV(model_SVM, parameters, n_jobs=-1)
GridSearch_SVM = GridSearch_SVM.fit(X, y)
print('Training the Model please be patient.')
print('\n')
print('GridSearch_SVM best score: \n',GridSearch_SVM.best_score_)
print('\n')
print('GridSearch_SVM best score: \n' ,GridSearch_SVM.best_params_)

#Fit the model with Train
model_SVM.fit(X_train, y_train)

pred = model_SVM.predict(X_test)
#Defining Confusion Matrix
confusion_matrix(pred, y_test)
print('Fit X, y with SVM Algorithm ',model_SVM.fit(X, y))
#model_SVM.fit(X, y)
#Export pkl
joblib.dump(GridSearch_SVM, "model_SVM.pkl")
#import pkl
model_SVModel = joblib.load("model_SVM.pkl")

y_preds = model_SVM.predict(X_test)
print('\n')
print('\n')
print('accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
print(classification_report(y_test, y_preds))

#Defining Range prediction from Sheet. 
tweet_preds = model_SVM.predict(tweets['content'])
#Defining Range to save the prediction
df_tweet_preds = tweets.copy()
df_tweet_preds['Polarity'] = tweet_preds
df_tweet_preds.shape

index = random.sample(range(tweet_preds.shape[0]), 20)
for text, sentiment in zip(df_tweet_preds.content[index],
                           df_tweet_preds.Polarity[index]):                 #Para SVM escribir en esta linea df_tweet_preds.Polarity[index]):
    print (sentiment, '===> ', text, '\n')