# -*- coding: utf-8 -*-
"""
Script to import our csv into txt

@author: Michael Rodriguez
"""

import csv

csvf = open("training.1600000.processed.noemoticon.csv")
dialect = csv.Sniffer().sniff(csvf.read(1024))
txt_rdr = csv.reader(csvf,dialect)

txt_wtr = open('f_training.txt','w')

for r in txt_rdr:
    line = r[0]+" "+r[-1]
    txt_wtr.write(line)
    txt_wtr.write("\n")
    print("Write line:",line)

print("\nWriting complete, closing file...")
txt_wtr.close()
print("Conversion complete. file name: f_training.txt")
