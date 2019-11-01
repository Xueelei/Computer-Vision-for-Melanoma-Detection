# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:38:54 2019

@author: dell
"""

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
                if row[i]==1:
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
                if row[i]==1:
                    labels.append(all_labels[i])
                    break
        
np.savez("../data/test_data.npz", files = np.array(files), labels = np.array(labels))