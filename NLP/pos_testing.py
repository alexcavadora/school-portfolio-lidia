import nltk
from nltk.corpus import cess_esp
from nltk import pos_tag
from nltk.tokenize import word_tokenize


sentence = "The quick brown fox read over the lazy dog. He's running very fast!"

tokens = word_tokenize(sentence)
tags = pos_tag(tokens)

print(tags)