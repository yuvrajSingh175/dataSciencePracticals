import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Load stopwords
stopWords = set(stopwords.words("english"))

# Read text file
with open("exp4text.txt", "r") as file:
    text = file.read()
    print(text)

# Tokenize words from text file
words = word_tokenize(text)
print(words)

# Create a frequency distribution dictionary for words
freq_dist = dict()

for word in words:
    if word.isalpha():  # Check if the word contains only alphabetic characters
        word = word.lower()
        if word in stopWords:
            continue
        if word in freq_dist:
            freq_dist[word] += 1
        else:
            freq_dist[word] = 1

# Print the frequency distribution
print(freq_dist)
