import json
import datetime
import os


def GTakeoutTelematics(folder):
	#Take input string, which is the file name, and load into the function.  This should be the default Google Takeouts name

	path = os.path.abspath(os.getcwd())
	Location = os.path.join(os.path.expanduser("~"),folder,"Takeout","Location History")

	print Location

	f = open(os.path.normpath(os.path.join(Location,'LocationHistory.json'))).read()

	data =json.loads(f)
	

	print datetime.datetime.fromtimestamp(float(int(loc['locations'][0]['timestampMs'])/1000))

if __name__ == '__main__':
	folder = raw_input("Type the name of the folder when your downloaded your Google Takeout data\n")
	GTakeoutTelematics(folder)