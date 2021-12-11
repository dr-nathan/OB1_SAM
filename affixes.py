#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:02:01 2021

@author: nathanvaartjes

The purpose of this script is to build a dictionnary of affixes and their log frequency per million (zipf).
Adapted from Jarmulowicz et al.(2002) NOTE: -ly not counted as suffix in this paper.
Necessary for the simulation for the Embedded Words task (Beyersmann 2020)

"""
import numpy as np
import pickle

    
affixes_totalcount_en={'tion_':122,
                       'al_':91,
                       'ial_':91,
                       'er_':85,
                       'y_':75,
                       'ment_':59, 
                       'ous_': 51,
                       'ious_':51,
                       'ant_':50, 
                       'ent_':50,
                       'an_':48,
                       'ian_':48,
                       'ar_':43,
                       'or_':43,
                       'ance_':30,
                       'ence_':30, 
                       'ity_':28,
                       'able_':24, 
                       'ible_':24,
                       'ate_':22,
                       'ful_':22,
                       'ive_':17,
                       'icev':16, 
                       'ise_':16, 
                       'ic_':13,
                       'en_':13, 
                       'ship_':11,
                       'ure_':11,
                       'ness_': 10, 
                       'ernv':8,
                       'age_':8, 
                       'ize_':8, 
                       'less_':6, 
                       'ism_':5,
                       'ary_':5, 
                       'thv':4,
                       'itev':3,
                       'istv':3,
                       'cracyv':2,
                       'ide_':1,
                       'hood_':1,
                       'ify_':1}

affixes_zipf={}

for i,(j,k) in enumerate(affixes_totalcount_en.items()):
    affixes_zipf[j]=np.log((k/24680)*1000000) #frequency per million (24680 words in text in total)


with open('Data/affixes_frequency_en.dat', 'wb')  as f:
    pickle.dump(affixes_zipf, f)

if __name__=="__main__":
    
    print(list(affixes_zipf.items()))
    