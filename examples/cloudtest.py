#!/usr/bin/env python2
# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 



# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# August 9, 2015

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

from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
from nltk.corpus import brown
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer

d = path.dirname(__file__)

# Text is coming from the NLTK Brown corpus; cr07
#test = brown.sents('cr07')
test = open('/home/linwood/Desktop/CIOPanelQues.txt','r')

#In case I want to do some scikit-learn work...not sure
vect = CountVectorizer(stop_words='english')

#Turnign list of lists from NLTK Brown Corpus cr07 into a single list
corpus = []
for l in test:
	for m in l:
		corpus.append(m)
#test1 = vect.fit_transform(test).todense()

# Taking list from NLTK Brown Corpus cr07 and turning it into a single string
text = (" ".join(map(str,corpus)))

# using stop word list from scikit-learn feature_extraction
#scikit is frozen, but the STOPWORDS function from wordclould can be customized
# just simply do stopwords = STOPWORDS.copy() and in the next line, stopwords.add("words") to customize
stopwords = stop_words.ENGLISH_STOP_WORDS

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
#tank_mask = imread('/home/linwood/Pictures/tank.png')
tank_mask = imread('/home/linwood/Pictures/soldier.jpg')

wc = WordCloud(background_color="white", max_words=1000, mask=tank_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file(path.join(d, "tank.png"))

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(tank_mask, cmap=plt.cm.gray)
plt.axis("off")
plt.show()