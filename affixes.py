#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:02:01 2021

@author: nathanvaartjes
"""
import numpy as np
import pickle

#list of affixes and their frequency. Adapted from Jarmulowicz et al.(2002).
#NOTE: -ly not counted as suffix in this paper.
    
affixes_totalcount={'-tion':122,
                    '-al':91,
                    '-ial':91,
                    '-er':85,
                    '-y':75,
                    '-ment':59, 
                    '-ous': 51,
                    '-ious':51,
                    '-ant':50, 
                    '-ent':50,
                    '-an':48,
                    '-ian':48,
                    '-ar':43,
                    '-or':43,
                    '-ance':30,
                    '-ence':30, 
                    '-ity':28,
                    '-able':24, 
                    '-ible':24,
                    '-ate':22,
                    '-ful':22,
                    '-ive':17,
                    '-ice':16, 
                    '-ise':16, 
                    '-ic':13,
                    '-en':13, 
                    '-ship':11,
                    '-ure':11,
                    '-ness': 10, 
                    '-ern':8,
                    '-age':8, 
                    '-ize':8, 
                    '-less':6, 
                    '-ism':5,
                    '-ary':5, 
                    '-th':4,
                    '-ite':3,
                    '-ist':3,
                    '-cracy':2,
                    '-ide':1,
                    '-hood':1,
                    '-ify':1}

affixes_zipf={}

for i,(j,k) in enumerate(affixes_totalcount.items()):
    affixes_zipf[j]=np.log((k/24680)*1000000) #frequency per million


print(list(affixes_zipf.items()))

with open('Data/affixes_frequency_en.dat', 'wb')  as f:
    pickle.dump(affixes_zipf, f)

