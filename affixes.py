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
import pandas as pd
import pickle
import copy
import string

prefixes_EN = pd.ExcelFile('Texts/MorphoLEX_en.xlsx').parse('All prefixes')
suffixes_EN = pd.ExcelFile('Texts/MorphoLEX_en.xlsx').parse('All suffixes')
prefixes_FR = pd.ExcelFile('Texts/Morpholex_FR.xlsx').parse('prefixes')
suffixes_FR = pd.ExcelFile('Texts/Morpholex_FR.xlsx').parse('suffixes')

prefixes_FR.rename(columns={"Unnamed: 0": "morpheme"}, inplace=True)
suffixes_FR.rename(columns={"Unnamed: 0": "morpheme"}, inplace=True)

prefixes_EN["morpheme"] = prefixes_EN["morpheme"].map(lambda L: L.strip("<"))
suffixes_EN["morpheme"] = suffixes_EN["morpheme"].map(lambda L: L.strip(">"))
prefixes_FR["morpheme"] = prefixes_FR["morpheme"].map(lambda L: L.strip("<"))

#suffixes FR is a bit more complicated
suffixes_FR["morpheme"] = suffixes_FR["morpheme"].map(lambda L: L.strip("<"))
suffixes_FR["morpheme"] = suffixes_FR["morpheme"].map(lambda L: L.strip(">"))
suffixes_FR["morpheme"] = suffixes_FR["morpheme"].map(lambda L: L.replace("[VB]", ""))
suffixes_FR["morpheme"] = suffixes_FR["morpheme"].map(lambda L: L.translate(str.maketrans("-"," ", string.punctuation)))


to_remove = []
for ix, row in suffixes_FR.iterrows():
    #ant/ent became antent after punct removal, so keep ant and add ent at the end of df
    if row["morpheme"] == 'antent':
        suffixes_FR.at[ix, "morpheme"] = 'ant'
        to_append = copy.deepcopy(row)
        to_append["morpheme"]='ent'
        
    if row["morpheme"] == '':
        to_remove.append(ix)

suffixes_FR.drop(index=to_remove, inplace=True)
suffixes_FR=suffixes_FR.append(to_append,ignore_index=True)


#old data (Jarmulowicz, 2002), kept for reference for now

# suffix_totalcount_en = {'tion': 122,
#                         'ion': 122,
#                         'al': 91,
#                         'ial': 91,
#                         'er': 85,
#                         'y': 75,
#                         'ment': 59,
#                         'ous': 51,
#                         'ious': 51,
#                         'ant': 50,
#                         'ent': 50,
#                         'an': 48,
#                         'ian': 48,
#                         'ar': 43,
#                         'or': 43,
#                         'ance': 30,
#                         'ence': 30,
#                         'ity': 28,
#                         'able': 24,
#                         'ible': 24,
#                         'ate': 22,
#                         'ful': 22,
#                         'ive': 17,
#                         'ice': 16,
#                         'ise': 16,
#                         'ic': 13,
#                         'en': 13,
#                         'ship': 11,
#                         'ure': 11,
#                         'ness': 10,
#                         'ern': 8,
#                         'age': 8,
#                         'ize': 8,
#                         'less': 6,
#                         'ism': 5,
#                         'ary': 5,
#                         'th': 4,
#                         'ite': 3,
#                         'ist': 3,
#                         'cracy': 2,
#                         'ide': 1,
#                         'hood': 1,
#                         'ify': 1}

# suffix_zipf = {}

# for i, (j, k) in enumerate(suffix_totalcount_en.items()):
#     # log10 of frequency per billion (24680 words in text in total)
#     suffix_zipf[j] = np.log10((k/24680)*1E9)


# with open('Data/suffix_frequency_en.dat', 'wb') as f:
#     pickle.dump(suffix_zipf, f)


names=['prefix_frequency_en',"suffix_frequency_en" ,"prefix_frequency_fr", "suffix_frequency_fr" ]

for ix, frame in enumerate([prefixes_EN, suffixes_EN, prefixes_FR, suffixes_FR]):
    
    #Morpholex EN has freq values as total count in HAL corpus. So, convert to Zipf
    if names[ix].endswith("_en"):
        
        freq_dict={}
        for ix2,  row in frame.iterrows():
            freq_dict[row["morpheme"]]=np.log10((row["HAL_freq"]/130000000)*1E9)
        
        with open(f'Data/{names[ix]}.dat', 'wb') as f:
            pickle.dump(freq_dict, f)
            
            
    # elif names[ix].endswith("_fr"):
        
    #     #TODO: scale of morpholex FR is still unclear! refer to emails, and correct when known
    #     freq_dict={}
    #     for ix2,  row in frame.iterrows():
    #         freq_dict[row["morpheme"]]=np.log10((row["HAL_freq"]/130000000)*1E9)
    
            
        

        
