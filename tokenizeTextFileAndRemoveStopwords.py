#*For TEXT INPUT FILE*
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Set stopwords language to English
stopWords = set(stopwords.words("english"))

# Read text file
with open("exp3text.txt", "r") as file:
    text = file.read()
    print(text)

# Tokenize words from text file
words = word_tokenize(text)
print(words)

# Remove stopwords from word list
text2 = []
stoplist = []
for word in words:
    word = word.lower()
    if word in stopWords:
        stoplist.append(word)
        continue
    text2.append(word)

# Words list without Stopwords
print(text2)

# List of stopwords we have omitted
print(stoplist)

# Tokenize sentences
sentences = sent_tokenize(text)
print(sentences)

# Number of sentences
print('No. of sentences:', len(sentences))

#*For TEXT INPUT STRING*
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Set stopwords language to English
stopWords = set(stopwords.words("english"))

# Input text as a string
text = "This is an example sentence. It includes some stop words."

# Print input text
print("Input Text:")
print(text)

# Tokenize words from input text
words = word_tokenize(text)
print("\nTokenized Words:")
print(words)

# Remove stopwords from word list
text2 = []
stoplist = []
for word in words:
    word = word.lower()
    if word in stopWords:
        stoplist.append(word)
        continue
    text2.append(word)

# Words list without Stopwords
print("\nWords without Stopwords:")
print(text2)

# List of stopwords we have omitted
print("\nList of Stopwords Omitted:")
print(stoplist)

# Tokenize sentences
sentences = sent_tokenize(text)
print("\nTokenized Sentences:")
print(sentences)

# Number of sentences
print('\nNo. of sentences:', len(sentences))
