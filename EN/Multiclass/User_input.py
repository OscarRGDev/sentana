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
from sklearn.exceptions import ConvergenceWarning

import warnings
#Remove DeprecationWarning and ConvergenceWarning Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)



tweets_df = pd.read_excel('/Users/oscar/Desktop/Sentiment/conneutros/Training_En.xlsx', header=0, encoding='iso8859_15')

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
parameters_SVM = {
    'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
    'tfidf__use_idf': (True, False)
}

model_Logistic_Regression = Pipeline([
    #Def stopword for model and Tf-idf  for the Tweets
    ('tfidf', TfidfVectorizer(stop_words=spa_stop)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag')))
])

#Logistic Regression:
parameters_Logistic_Regression = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}
#MultinomialNB:

Model_MultinomialNB = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words=spa_stop,
                            tokenizer = tokenization,
                            analyzer = 'word',
                            lowercase=True)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

parameters_MultinomialNB = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }

print('Training the Model please be patient.')
GridSearch_SVM = GridSearchCV(model_SVM, parameters_SVM, n_jobs=-1)
GridSearch_SVM = GridSearch_SVM.fit(X, y)

GridSearch_Log_Reg = GridSearchCV(model_Logistic_Regression, parameters_Logistic_Regression, n_jobs=-1)
GridSearch_Log_Reg = GridSearch_Log_Reg.fit(X, y)


GridSearch_MultinomialNB = GridSearchCV(Model_MultinomialNB, cv=10, param_grid=parameters_MultinomialNB, verbose=1)
GridSearch_MultinomialNB= GridSearch_MultinomialNB.fit(X_train,y_train)


model_SVM_best_parameters =Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])

#Fit the model with Train
model_SVM_best_parameters.fit(X_train, y_train)

pred_SVM = model_SVM_best_parameters.predict(X_test)


model_Logistic_Regression.fit(X_train, y_train)

pred_Logistic_Regression = model_Logistic_Regression.predict(X_test)

#Export pkl
joblib.dump(GridSearch_SVM, "model_User_Tweet_Input_SVM.pkl")
joblib.dump(GridSearch_Log_Reg, "model_User_Tweet_Input_Logistic_Regression.pkl")
joblib.dump(GridSearch_MultinomialNB, "model_User_Tweet_Input_MultinomialNB.pkl")

#import pkl
model_Logistic_Regression = joblib.load("model_User_Tweet_Input_Logistic_Regression.pkl")
model_SVM_best_parameters = joblib.load("model_User_Tweet_Input_SVM.pkl")
model_MultinomialNB = joblib.load("model_User_Tweet_Input_MultinomialNB.pkl")



tweet = input("User input: ")
print('Result with SVM Algorithm', model_SVM_best_parameters.predict([tweet])[0])
print('Result with Logistic Regression Algorithm', model_Logistic_Regression.predict([tweet])[0])
print('Result with MultinomialNB Algorithm', model_MultinomialNB.predict([tweet])[0])
