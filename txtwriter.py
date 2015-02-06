# -*- coding: utf-8 -*-
"""
Script to import our csv into txt
    Update: now creates two files, one for positive tweets one for negative
@author: Michael Rodriguez
"""

import csv

csvf = open("training.1600000.processed.noemoticon.csv")
dialect = csv.Sniffer().sniff(csvf.read(1024))
txt_rdr = csv.reader(csvf,dialect)

pos = open('ptweets.txt','w')
neg = open('negtweets.txt','w')

for r in txt_rdr:
    line = r[0]+" "+r[-1]
    if r[0] == '0':
        neg.write(line)
        neg.write("\n")
        print("Writing neg line: ", line)
    else:
        pos.write(line)
        pos.write("\n")
        print("Writing pos line: ",line)
        
print("\nWriting complete, closing files...")
neg.close()
pos.close()
print("Conversion complete.")
