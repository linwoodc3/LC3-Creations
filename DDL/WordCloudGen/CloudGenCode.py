#!/usr/bin/env python2
# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 



# Text analytics on audience submitted panel questions.

# August 26, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# External Help or online resources used
###############################################################################

# http://stackoverflow.com/questions/26126442/combining-text-stemming-and-removal-of-punctuation-in-nltk-and-scikit-learn


"""
Masked wordcloud
================
Using a mask you can generate wordclouds in arbitrary shapes.
"""

###############################################################################
# Imports
###############################################################################

import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
from nltk.corpus import brown
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

path = os.path.abspath(os.getcwd())

###############################################################################
# Preprocessing, assuming you are starting from excel sheet
###############################################################################

# Load in the excel questions file from www.slido.com "After the event" exported excel file; preprocess, encode, and store as text file


import pandas as pd

df = pd.read_excel(os.path.normpath(os.path.join(path,'Data','DoDIIS Worldwide questions.xls')))

a= [l for l in df.loc[:]['Text']]
new = [line.encode('utf-8') for line in a]
cheese = repr(new)
b = (cheese.decode('unicode_escape').encode('ascii','ignore'))

import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
wordnet_tags = ['n', 'v', 'a', 's', 'r']

lemmatizer = WordNetLemmatizer()


def lemmatize(token,tag):
        if tag[0].lower() in ['n', 'v', 'a', 's', 'r']:
                return lemmatizer.lemmatize(token,tag[0].lower())
        return token

# punt = re.sub('[&,\]\[]|\''," ", b)
fumble = re.sub('IC ITE', 'ICITE',b)
punt = re.sub('[&,\]\[]|\''," ", fumble)
tagged_corpus = pos_tag(word_tokenize(punt))
slim = [lemmatize(token,tag) for token,tag in tagged_corpus]
newslim =  (" ".join(map(str,slim)))

data = open(os.path.join(path,'Data','LemCIOQues.txt'),'w+')
data.write(newslim)

###############################################################################
# Major Functions
###############################################################################

with open(os.path.normpath(os.path.join(path,'Data','LemCIOQues.txt'))) as inp:
	data = list(inp)
text = data[0]

# using stop word list from scikit-learn feature_extraction
#scikit is frozen, but the STOPWORDS function from wordclould can be customized
# just simply do stopwords = STOPWORDS.copy() and in the next line, stopwords.add("words") to customize
newstop = set([l for l in stop_words.ENGLISH_STOP_WORDS])
stopwords = newstop
stopwords.add("DIA")
stopwords.add("CIO")
stopwords.add("use")
stopwords.add("s")
stopwords.add("IC ")
stopwords.add("IC")
stopwords.add("ITE")
stopwords.add("t")
stopwords.add("ESITE")

print stopwords


# read the mask image
custom_mask = imread(os.path.normpath((os.path.join(path,'Pictures for Mask','soldier.png'))))

wc = WordCloud(background_color="white", max_words=150, mask=custom_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file(os.path.normpath((os.path.join(path,'Output','cloudsoldier.png'))))

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(custom_mask, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
