import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Read the text file
with open("exp5text.txt", "r") as file:
    text = file.read()
    print(text)

# Tokenize words from the text
words = word_tokenize(text)
print(words)

# Create a frequency distribution dictionary for words
freq_dist = dict()

# Load stopwords
stopWords = set(stopwords.words("english"))

for word in words:
    if word.isalpha():  # Check if the word contains only alphabetic characters
        word = word.lower()
        if word in stopWords:  # Should be "stopWords" instead of "stopwords"
            continue
        if word in freq_dist:
            freq_dist[word] += 1
        else:
            freq_dist[word] = 1
