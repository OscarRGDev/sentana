from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB                          # GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#tweets_df = pd.read_excel('Desktop/bla_4_sin0.xlsx' , header=0, encoding='iso8859_15')
#tweets = pd.read_excel('Downloads/Analisis_Sentimientos_Twitter-master/ficheros/TweetsConTopic/tweets_spainGeo_topic_nadia.xlsx', header=0, encoding='iso8859_15')

tweets_df = pd.read_excel('/Users/oscar/Desktop/Sentiment/conneutros/Training_Es.xlsx', header=0, encoding='iso8859_15')
tweets = pd.read_excel('/Users/oscar/Desktop/Sentiment/conneutros/Test_Es.xlsx', header=0, encoding='iso8859_15')

X = []
for i in range(tweets_df.shape[0]):
    X.append(tweets_df.iloc[i][0])
y = np.array(tweets_df["polarity_bin"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = Pipeline([('vectorizer', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])


from sklearn.grid_search import GridSearchCV
parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
               'tfidf__use_idf': (True, False)}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X, y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)


model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])


model.fit(X_train, y_train)

pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred, y_test)

model.fit(X, y)

from sklearn.externals import joblib
joblib.dump(gs_clf_svm, "model_SVM.pkl")

model = joblib.load("model_SVM.pkl")

y_preds = model.predict(X_test)

print('accuracy score: ',accuracy_score(y_test, y_preds))
print('\n')
print('confusion matrix: \n',confusion_matrix(y_test,y_preds))
print('\n')
print(classification_report(y_test, y_preds))

tweet_preds = model.predict(tweets['content'])

df_tweet_preds = tweets.copy()
df_tweet_preds['Polarity'] = tweet_preds
df_tweet_preds.shape

index = random.sample(range(tweet_preds.shape[0]), 20)
for text, sentiment in zip(df_tweet_preds.content[index],
                           df_tweet_preds.Polarity[index]):                 #Para SVM escribir en esta linea df_tweet_preds.Polarity[index]):
    print (sentiment, '--', text, '\n')