from cProfile import label
from turtle import shape
import nltk #Natural Language Tool Kit (NLTK) is a Python library to make programs that work with natural language.
nltk.download('punkt')
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import random
import json
import tensorflow as tf
import tflearn

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = [] 
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words  if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

traning = numpy.array(training)
output = numpy.array(output)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None,len(traning[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

model.fit(traning, output, n_epoch=1000, batch_size = 8, show_metric=True)
model.save("model.tflearn")

