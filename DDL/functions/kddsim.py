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

from os import walk
import os
import subprocess
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import time
from guppy import hpy
from lemmatize import lemmatize

wordnet_tags = ['n','v']


###############################################################################
# Admin functions
###############################################################################

path        = os.path.abspath(os.getcwd())
#/home/linwood/projects/LC3-Creations/examples/KDDsample
TESTDIR     = os.path.normpath(os.path.join(os.path.expanduser("~"),"projects","LC3-Creations", "examples","KDDsample"))
#INPUT_DIR  = os.path.normpath(os.path.join(os.path.expanduser("~"),"Desktop","KDD 2015", "docs"))

###############################################################################
# Main Function
###############################################################################

# This code will iterate over files and extract text from the PDF; each document is an item in the list

corpus = []

start_time = time.time()
def extractPDFtext(fileName):
    print fileName
    print os.path.normpath(os.path.join(TESTDIR,fileName))
    corpus.append(subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))]))


    '''
    if len(corpus) == 10:
        print corpus[8]

    else:
        pass
    '''

###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
    for dirName, subdirList, fileList in walk(TESTDIR):
        for fileName in fileList:
            if fileName.startswith('p') and fileName.endswith('.pdf'):
            	extractPDFtext(fileName)

    lemmatize()
    h = hpy()
    print h.heap()
    print len(corpus)
    print ("---%s seconds ---" % (time.time() - start_time))
