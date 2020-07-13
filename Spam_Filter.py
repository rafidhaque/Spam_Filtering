# All The Imports

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure(figsize=(7, 5))

import nltk

nltk.download('punkt')

from nltk.stem import PorterStemmer

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# Getting data through pandas
my_data = pd.read_csv("spam.csv", delimiter=',', encoding='latin1')

# Printing the length of each sentence.

lis = my_data['v2'].tolist()
# for sentence in lis:
#     print(len(sentence))

# Calculating the number of messages in each class.

no_of_hams = 0
no_of_spams = 0
missing_data = 0

lis1 = my_data['v1'].tolist()
lis2 = my_data['v2'].tolist()

for label, data in zip(lis1, lis2):
    if len(data) != 0:
        if label == 'ham':
            no_of_hams += 1
        else:
            no_of_spams += 1
    else:
        missing_data += 1

# print(no_of_hams)
# print(no_of_spams)
# print(missing_data)

# Plotting number of hams and spams into a bar diagram.

labels = ['Spam', 'Ham']
count_of_data = [no_of_spams, no_of_hams]
positions = [0, 1]

plt.bar(positions, count_of_data)

plt.show()

# Tokenizing the data.
# Splitting Sentences into Words' lists.
# Also removing all punctuation marks.

list_of_tokens = []
list_of_tokenized_sentences = []

tokenizer = nltk.RegexpTokenizer(r"\w+")

for sentence in lis2:
    new_words = tokenizer.tokenize(sentence)
    list_of_tokenized_sentences.append(new_words)
    list_of_tokens += new_words

for i in list_of_tokens:
    if i == '.':
        list_of_tokens.remove(i)

# print(list_of_tokens)

# Stemming the data.
# Finding morphological variants of a base word.
# Also turning all the words lowercased.

ps = PorterStemmer()
stemmed_list = []
stemmed_sentences_list = []

for token in list_of_tokens:
    stemmed_word = ps.stem(token.lower())
    if len(stemmed_word) == 1:
        continue
    stemmed_list.append(stemmed_word)

for word_list in list_of_tokenized_sentences:
    temporary_list = []
    for word in word_list:
        word = ps.stem(word)
        word = word.lower()
        temporary_list.append(word)
    stemmed_sentences_list.append(temporary_list)

# print(stemmed_list)
# print(stemmed_sentences_list)

# Using stop words set remove stop words

stop_set = {'then', 'are', 'through', 'yourself', 'each', 'once', 'on', 'further', 'after', 'his', 'some', 'yourselves',
            'don', 'from', 'or', 'the', 'her', 'that', 'having', 'and', 'if', 'more', 'she', 'they', 'off', 'will',
            'those', 'under', 'whom', 'any', 'were', 'in', 'was', 'into', 's', 'did', 'not', 't', 'had', 'as', 'too',
            'themselves', 'itself', 'being', 'a', 'during', 'other', 'have', 'until', 'how', 'there', 'am', 'which',
            'same', 'him', 'does', 'of', 'out', 'again', 'up', 'own', 'me', 'but', 'what', 'himself', 'an', 'no',
            'than', 'our', 'so', 'your', 'between', 'their', 'is', 'below', 'myself', 'only', 'with', 'over', 'be',
            'against', 'i', 'all', 'at', 'these', 'when', 'this', 'my', 'hers', 'about', 'before', 'he', 'by', 'most',
            'to', 'ourselves', 'while', 'we', 'should', 'where', 'here', 'why', 'who', 'both', 'few', 'because', 'ours',
            'above', 'herself', 'has', 'its', 'it', 'down', 'now', 'such', 'you', 'nor', 'just', 'do', 'very', 'been',
            'for', 'them', 'can', 'yours', 'doing', 'theirs'}

stemmed_list = set(stemmed_list)
stemmed_list = list((stemmed_list.difference(stop_set)))

# print(stemmed_list)

# Counting the frequency of each word in the document.

word_frequency_dictionary = {}
for word in stemmed_list:
    word_frequency_dictionary[word] = 0

for word_list in stemmed_sentences_list:
    for word in word_list:
        if word in stemmed_list:
            word_frequency_dictionary[word] += 1

# print(word_frequency_dictionary)

# Finding top 100 most frequent word.

sorted_list_of_word_frequency = sorted(word_frequency_dictionary.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
top_100_frequent_words = sorted_list_of_word_frequency[:100]

top_100_frequent_word_list = []
for key, value in top_100_frequent_words:
    top_100_frequent_word_list.append(key)


# print(top_100_frequent_word_list)


# counts words in a list

def word_count(lis):
    counts = dict()
    for word in lis:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


# Creating a Feature Vector matching sentences with top 100 words.

df = pd.DataFrame(0, index=lis2, columns=top_100_frequent_word_list)
i = 0

for sentence in stemmed_sentences_list:
    wordcount = word_count(sentence)
    for key, value in wordcount.items():
        if key in top_100_frequent_word_list:
            df.at[lis[i], key] = value
    i += 1

# creating feature matrix

features_matrix = {}
for word in top_100_frequent_word_list:
    lis = df[word].tolist()
    features_matrix[word] = lis

states = []
for i in lis1:
    if i == 'ham':
        states.append(0)
    else:
        states.append(1)

# Creating X and y

X = pd.DataFrame(features_matrix).to_numpy()
y = np.array(states)

# recall, precision and accuracy

recall, precision, accuracy, score = 0, 0, 0, 0


def compute_accuracy(tp, tn, fn, fp):
    return ((tp + tn) * 100) / float(tp + tn + fn + fp)


def compute_recall(tp, fn):
    return (tp * 100) / float(tp + fn)


def compute_precision(tp, fp):
    return (tp * 100) / float(tp + fp)


# K Folding

kf = KFold(n_splits=10)

for train_index, text_index in kf.split(X):
    X_train, X_test = X[train_index], X[text_index]
    y_train, y_test = y[train_index], y[text_index]

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    score += (mnb.score(X_test, y_test))

    y_prediction = mnb.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()
    accuracy += (compute_accuracy(tp, tn, fn, fp))
    recall += (compute_recall(tp, fn))
    precision += (compute_precision(tp, fp))

# finding average score, recall, precision and accuracy.

score = score / 10
accuracy = accuracy / 10
recall = recall / 10
precision = precision / 10

print(score)
print(accuracy)
print(recall)
print(precision)