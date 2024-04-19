import SimpleITK as sitk
import os, glob
import json
import numpy as np
import re

def extract_number(filename):
    # Use regular expression to find the number in the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1 

filelist =  sorted(glob.glob('./norm_data/*.nii.gz'),  key=extract_number)
seglist =  sorted(glob.glob('./norm_label/*.nii.gz'),  key=extract_number)

keyword = 'train'
dictout = {keyword:[]}
for i in range (0, len(filelist)):
	smalldict = {}
	print ('./Brains'+ filelist[i][1:])
	print ('./Brains' + seglist[i][1:])
	for j in range (0, len(filelist)):
		if(i!=j):
			smalldict['Source'] = './Brains'+ filelist[i][1:]
			smalldict ['Target'] = './Brains' + filelist[j][1:]
			smalldict['Source_label'] = './Brains' + seglist[i][1:]
			smalldict ['Target_label'] ='./Brains' + seglist[j][1:]
			dictout[keyword].append(smalldict)

savefilename = './data'+ '.json'
with open(savefilename, 'w') as fp:
	json.dump(dictout, fp)