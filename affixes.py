#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:02:01 2021

@author: nathanvaartjes

The purpose of this script is to build a dictionnary of affixes and their log frequency per million (zipf).
This dictionary is then pickled to later be fetched when needed.
Adapted from Jarmulowicz et al.(2002)(DOI: 10.1006/brln.2001.2517 ) NOTE: -ly not counted as suffix in this paper.
Necessary for the simulation for the Embedded Words task (Beyersmann 2020)

"""
import numpy as np
import pickle

#TODO: in order to implement prefixes, get data on frequency from scientific literature, 
# and implement following the suffix structure hereunder    


suffix_totalcount_en={'tion':122,
                       'al':91,
                       'ial':91,
                       'er':85,
                       'y':75,
                       'ment':59, 
                       'ous': 51,
                       'ious':51,
                       'ant':50, 
                       'ent':50,
                       'an':48,
                       'ian':48,
                       'ar':43,
                       'or':43,
                       'ance':30,
                       'ence':30, 
                       'ity':28,
                       'able':24, 
                       'ible':24,
                       'ate':22,
                       'ful':22,
                       'ive':17,
                       'ice':16, 
                       'ise':16, 
                       'ic':13,
                       'en':13, 
                       'ship':11,
                       'ure':11,
                       'ness': 10, 
                       'ern':8,
                       'age':8, 
                       'ize':8, 
                       'less':6, 
                       'ism':5,
                       'ary':5, 
                       'th':4,
                       'ite':3,
                       'ist':3,
                       'cracy':2,
                       'ide':1,
                       'hood':1,
                       'ify':1}

suffix_zipf={}

for i,(j,k) in enumerate(suffix_totalcount_en.items()):
    suffix_zipf[j]=np.log((k/24680)*1000000) #log of frequency per million (24680 words in text in total)


with open('Data/suffix_frequency_en.dat', 'wb')  as f:
    pickle.dump(suffix_zipf, f)
    

#insert prefix code here

if __name__=="__main__":
    
    print(list(suffix_zipf.items()))
    