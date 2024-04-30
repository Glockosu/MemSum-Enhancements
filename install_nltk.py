import nltk
from nltk.tokenize import sent_tokenize

# Test sentence tokenization
text = "This is a test. This is only a test."
sentences = sent_tokenize(text)
print(sentences)