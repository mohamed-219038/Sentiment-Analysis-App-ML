import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocessing_step(text):
  # lower text
  text = text.lower()
  # remove any special character
  text = re.sub(r"[^a-zA-Z0-9]", " ", text)
  # tokenization 'i love you ?????!!!' -> 'i', 'love', 'you' , '?', '?'
  token = word_tokenize(text)
  # stop words
  stop_word = stopwords.words("english")
  token_without_stop =  [word for word in token if word not in stop_word]

  # steming porter stemmer(fast with low accuracy) / lemetization(slow but accurate)
  ps = PorterStemmer()
  token_stem = [ps.stem(word) for word in token_without_stop]
  return " ".join(token_stem)