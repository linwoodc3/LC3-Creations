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
import sys
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import time
#from guppy import hpy
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
    #print os.path.normpath(os.path.join(TESTDIR,fileName))
    # corpus[str(fileName)]=subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))])

    # a = ((repr(subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))]).encode('utf-8'))).decode('unicode_escape').encode('ascii','ignore'))
    a = unicode(subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))]),errors='ignore')

    corpus['Text']=a
    print corpus
    print sys.getsizeof(corpus)
    print len(corpus)
    return

    # This line extracts the title from the pdf file
    Title = unicode(re.findall("^[^\\n]*",a)[0],errors='ignore')

    # This line of code returns the abstract, authors, universities
    print "\n %s" % unicode(re.findall(r'\n\n([^]]*)\n\n',a[:2500])[0], errors='ignore')

    # Get entities from the document, switch up the \n's and append to list
    entities = [i for i in re.findall(r'\n\n(.+?)\n',snip)
]


    print "\nFinished processing %s" % Title

    #corpus[str(fileName)]=PDFcut.convert(str(os.path.normpath(os.path.join(TESTDIR,fileName))))

    # Removes quotes and escapes from title
    # re.sub(r'[\"\\]'," ",Title)


    '''
    I'm going to store the lemmatized tokens, stemmed tokens, tfidf matrix, Title, filename,
    year, author, extracted entities, tfidf of extracted entities,
    '''

    # Creating a dictionary within another dictionary to store information about my documents
    # [str(Title)]['Details']={}
    # corpus[str(Title)][''] =
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
    #h = hpy()
    #print h.heap()
    print len(corpus)
    print ("---%s seconds ---" % (time.time() - start_time))



###############################################################################
# Graveyard for stuff that didn't work
###############################################################################

'''
#Extract text from pdf and get rid of unicode errors
a = subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))])
cheese = repr(a)
b = (cheese.decode('unicode_escape').encode('ascii','ignore'))


#Extract the title
Title = ((re.search(r"(.*?)\w*\\n",cheese).group(0)).decode('unicode_escape').encode('ascii','ignore')).rstrip('\n')).lstrip('\'')

'''
