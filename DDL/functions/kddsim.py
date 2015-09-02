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
import json
import PDFcut
import re


wordnet_tags = ['n', 'v', 'a', 's', 'r']


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

corpus = {}

start_time = time.time()
def extractPDFtext(fileName):
    print fileName
    #print os.path.normpath(os.path.join(TESTDIR,fileName))
    # corpus[str(fileName)]=subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))])
    #a = ((repr(subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))]).encode('utf-8'))).decode('unicode_escape').encode('ascii','ignore'))
    a = subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))])
    cheese = repr(a)
    b = (cheese.decode('unicode_escape').encode('ascii','ignore'))
    #corpus[str(fileName)]=PDFcut.convert(str(os.path.normpath(os.path.join(TESTDIR,fileName))))
    print re.search(r"(.*?)\w*\\",cheese).group(0)

    '''
    if len(corpus) == 1:
        print corpus[str(fileName)]

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

    print corpus
    h = hpy()
    print h.heap()
    print len(corpus)
    print ("---%s seconds ---" % (time.time() - start_time))
