import nltk
import numpy as np
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
from nltk.stem import WordNetLemmatizer 
# nltk.download("punkt")# punkt is an nltk package which has pre trained tokenizer which makes  .word_tokenizer function work ,is downloaded once and can be used anytime afterwards.
from nltk.stem.porter import PorterStemmer #from different types of stemmers, portersetmmer is chosen.
def tokenize(sentence):
    return nltk.word_tokenize(sentence)



def lemmatize(word):
    
    lemm=WordNetLemmatizer()
    lemm_word=lemm.lemmatize(word)
    return lemm_word.lower()



def stem(word):
    stemmer=PorterStemmer()
    return stemmer.stem(word.lower())#first lower the word then stemmerize it
def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx , w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag

