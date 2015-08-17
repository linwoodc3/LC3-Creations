# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
__author__ =  'Linwood Creekmore'


# Code to publish at KDD 2016

# August 13, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/



###############################################################################
# Imports
###############################################################################

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag




###############################################################################
# Admin
###############################################################################
lemmatizer = WordNetLemmatizer()
wordnet_tags = ['n','v','a','s','r']

###############################################################################
# Main Function
###############################################################################

tagged_corpus = [pos_tag(word_tokenize(corpus))

def lemmatize(token,tag):
    if tag[0].lower() in ['n','v','a','s','r']:
        return lemmatizer.lemmatize(token,tag[0].lower())
    return token


final = [[lemmatize(token, tag) for token,tag in tagged_corpus]]
cleaned = (" ".join(map(str,final[0])))