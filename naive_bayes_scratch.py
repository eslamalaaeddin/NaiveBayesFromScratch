# # Coding a spam classifier with naive Bayes
# ### 1. Imports and pre-processing data
# We load the data into a Pandas DataFrame, and then preprocess it by adding a string with the (non-repeated) lowercase words in the email.


import numpy as np

import pandas as pd

emails = pd.read_csv('emails.csv')

print(emails[:10])


def process_email(email):
    email = email.lower()
    return list(set(email.split()))


emails['words'] = emails['text'].apply(process_email)

print(emails[:10])

num_emails = len(emails)
num_spam = sum(emails['spam'])

print("Number of emails:", num_emails)
print("Number of spam emails:", num_spam)
print()

# Calculating the prior probability that an email is spam
print("Probability of spam:", num_spam / num_emails)

# ### 2. Training a naive Bayes model
# Our plan is to write a dictionary, and in this dictionary record every word, and its pair of occurrences in spam and ham

model = {}

# Training process
# Iterate over DataFrame rows as (index, Series) pairs.
for index, email in emails.iterrows():
    for word in email['words']:
        if word not in model:
            model[word] = {'spam': 1, 'ham': 1}
        if word in model:
            if email['spam']:
                model[word]['spam'] += 1
            else:
                model[word]['ham'] += 1

print(model['lottery'])
print(model['sale'])


##### 3. Using the model to make predictions


def get_word_probability_being_spam(word):
    word = word.lower()
    num_spam_with_word = model[word]['spam']  # Get word count in all spam emails
    num_ham_with_word = model[word]['ham']  # Get word count in all ham emails
    return 1.0 * num_spam_with_word / (num_spam_with_word + num_ham_with_word)


print(get_word_probability_being_spam('lottery'))
print(get_word_probability_being_spam('sale'))


def predict_naive_bayes(email):
    # Calculate total number of emails, spam emails, ham emails
    total = len(emails)
    num_spam = sum(emails['spam'])
    num_ham = total - num_spam

    # Turn each email into its words in lowercase
    email = email.lower()
    words = set(email.split())

    prob_of_mail_being_spam = [1.0]
    prob_of_mail_being_ham = [1.0]
    # For each word,
    for word in words:
        if word in model:
            prob_of_mail_being_spam.append(model[word]['spam'] / num_spam * total)
            prob_of_mail_being_ham.append(model[word]['ham'] / num_ham * total)
    prod_spams = np.prod(prob_of_mail_being_spam) * num_spam
    prod_hams = np.prod(prob_of_mail_being_ham) * num_ham
    return prod_spams / (prod_spams + prod_hams)


print("------------------------------------------------------")
print(predict_naive_bayes('lottery sale'))

print(predict_naive_bayes('Hi mom how are you'))

print(predict_naive_bayes('Hi MOM how aRe yoU afdjsaklfsdhgjasdhfjklsd'))

print(predict_naive_bayes('meet me at the lobby of the hotel at nine am'))

print(predict_naive_bayes('enter the lottery to win three million dollars'))

print(predict_naive_bayes('buy cheap lottery easy money now'))

print(predict_naive_bayes('Grokking Machine Learning by Luis Serrano'))

print(predict_naive_bayes('asdfgh'))
