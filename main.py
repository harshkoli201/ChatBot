import nltk #Natural Language Tool Kit (NLTK) is a Python library to make programs that work with natural language.
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import random
import json
import tensorflow
# import tflearn

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])