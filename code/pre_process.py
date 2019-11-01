# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:38:54 2019

@author: dell
"""


from PIL import Image
import os, sys

path = "../data/ISIC_2019_Training_Input/"
dirs = os.listdir( path )

def resize():
    cnt = 0
    for item in dirs:
        if os.path.isfile(path+item):
            print(cnt)
            cnt += 1
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((500, 383), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

#resize()

import numpy as np

import csv

# generate train data
files = []
labels = []


with open('../data/ISIC_2019_Training_GroundTruth.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cnt = 0
    for row in csv_reader:
        all_labels = row
        break
    for row in csv_reader:
        cnt += 1
        if cnt % 100 == 0:
            files.append(row[0]+'.jpg')
            for i in range(1, len(row)):
                if row[i]=='1.0':
                    labels.append(all_labels[i])
                    break
     
np.savez("../data/train_data.npz", files = np.array(files), labels = np.array(labels))


# generate test data
files = []
labels = []
with open('../data/ISIC_2019_Training_GroundTruth.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cnt = 0
    for row in csv_reader:
        all_labels = row
        break
    for row in csv_reader:
        cnt += 1
        if cnt % 999 == 0:
            files.append(row[0]+'.jpg')
            for i in range(1, len(row)):
                if row[i]=='1.0':
                    labels.append(all_labels[i])
                    break
        
np.savez("../data/test_data.npz", files = np.array(files), labels = np.array(labels))