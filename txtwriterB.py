# -*- coding: utf-8 -*-
"""
Script to import our csv into txt
Version B: scores not there categorized into two files with no score
@author: Michael Rodriguez
"""

import csv

csvf = open("training.1600000.processed.noemoticon.csv")
dialect = csv.Sniffer().sniff(csvf.read(1024))
txt_rdr = csv.reader(csvf,dialect)

pos = open('ptweets.txt','w')
neg = open('negtweets.txt','w')

for r in txt_rdr:
    if r[0] == '0':
        neg.write(r[-1]+'\n')
        print("Writing neg line: ", r[-1])
    else:
        pos.write(r[-1]+'\n')
        print("Writing pos line: ", r[-1])
        
print("\nWriting complete, closing files...")
neg.close()
pos.close()
print("Conversion complete.")
