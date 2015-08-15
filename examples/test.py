# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 



# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# August 9, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################


import os
from scipy.misc import imread


###############################################################################
# File Paths
###############################################################################

path = os.path.abspath(os.getcwd())

###############################################################################
# Helper Functions
###############################################################################

###############################################################################
# Main Functions
###############################################################################

def dragndrop(f):

	image = os.path.normpath(f)
	print (image)
	#mask = imread(image)


###############################################################################
# 'Main' Function
############################################################################### 

if __name__ == '__main__':
	f = raw_input('Drag and drop the file you want to use as a mask\n>')

	dragndrop(f)