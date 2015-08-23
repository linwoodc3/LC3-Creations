# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
__author__ =  'Linwood Creekmore'


# Code to publish at KDD 2016

# August 16, 2015

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
tagged_corpus = [pos_tag(word_tokenize(corpus))]
wordnet_tags = ['n','v','a','s','r']

###############################################################################
# Main Function
###############################################################################

'''This function will lem and stem the document, ultimately reducing the dimensions of the processed text file. '''



def lemmatize(token,tag):
    if tag[0].lower() in ['n','v','a','s','r']:
        return lemmatizer.lemmatize(token,tag[0].lower())
    return token






###############################################################################
# 'Main' Function
###############################################################################

if __name__ == '__main__':
    final = [[lemmatize(token, tag) for token,tag in tagged_corpus]]
    cleaned = (" ".join(map(str,final[0])))