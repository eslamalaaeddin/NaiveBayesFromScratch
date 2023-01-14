import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# https://binitaregmi.medium.com/spam-classification-using-naive-bayes-algorithm-3e263061a3b0

start = time.perf_counter()

data = pd.read_csv('emails.csv')
print(data.shape)

X_train, X_test, Y_train, Y_test = train_test_split(data['text'],
                                                    data['spam'],
                                                    train_size=.8,
                                                    test_size=.2,
                                                    random_state=0)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# extract features
vectorizer = CountVectorizer(ngram_range=(1, 1)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)
print(X_train_vectorized.toarray().shape)

"""
The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).
 The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
"""
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)

predictions = model.predict(vectorizer.transform(X_test))
print(accuracy_score(Y_test, predictions) * 100)

# record end time
end = time.perf_counter()

# find elapsed time in seconds
ms = (end - start)
print(f"Elapsed {ms:.03f} secs.")


print(model.predict_proba(vectorizer.transform([
    'lottery sale',
    'Hi mom how are you',
    'Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd',
    'meet me at the lobby of the hotel at nine am',
    'enter the lottery to win three million dollars',
    'buy cheap lottery easy money now',
    'Grokking Machine Learning by Luis Serrano sale',
    'asdfgh'
])))
