#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:09:16 2021

@author: nathanvaartjes
"""
import pandas as pd

EW=pd.read_csv('Stimuli/EmbeddedWords_stimuli_all_csv.csv' )

txt_out=[]
txt_out.extend(EW['prime'])
for i in range(len(EW['all'])):
    txt_out.append(EW['all'][i].lower())

with open('Texts/EmbeddedWords_freq_pred.txt', 'w') as f:
    for i in range(len(txt_out)):
        f.write(txt_out[i]+'\n')