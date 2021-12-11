#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:23:34 2021

@author: nathanvaartjes

This script creates a pickle file that contains the words of the specific task, 
appends their relatives frequencies and predictabilities, as calculated by the SUBTLEX or other relevant resource. 
Also appends 200 most common words of the language to the list 
It puts the generated .dat file in /Data
"""
import chardet
from parameters import return_params
import numpy as np
import pickle
from read_saccade_data import get_words

pm=return_params()


#NV: This script only builds freq_pred file for task specified as task_to_run in parameters.py
task = pm.task_to_run 
 
#NV: get appropriate freq dictionary (SUBTLEX-UK for english, Lexicon Project for french,...). Automatically detects encoding via Chardet and uses the value during import. Due to Chardet, its a bit slow however.
if pm.language=='english':
    freqlist_arrays = np.genfromtxt("Texts/SUBTLEX_UK.txt", dtype=[('Spelling','U30'),('FreqCount','f4'),('LogFreqZipf','f4')],
                                    usecols = (0,1,5),encoding=chardet.detect(open("Texts/SUBTLEX_UK.txt","rb").read())['encoding'] , skip_header=1, delimiter="\t", filling_values = 0)
    lang='en'
elif pm.language=='french':
    freqlist_arrays = np.genfromtxt("Texts/French_Lexicon_Project.txt", dtype=[('Word','U30'),('cfreqmovies','f4'), ('lcfreqmovies','f4'),('cfreqbooks','f4'), ('lcfreqbooks','f4')],
                                usecols = (0,7,8,9,10),encoding=chardet.detect(open("Texts/French_Lexicon_Project.txt","rb").read())['encoding'] , skip_header=1, delimiter="\t", filling_values = 0)
    lang='fr'
elif pm.language=='german':
    freqlist_arrays = np.genfromtxt("Texts/SUBTLEX_DE.txt", dtype=[('Word','U30'),('FreqCount','i4'), ('CUMfreqcount','i4'),('Subtlex','f4'), ('lgSubtlex','f4'), ('lgGoogle','f4')],
                                usecols = (0,1,3,4,5,9) , encoding=chardet.detect(open("Texts/SUBTLEX_DE.txt","rb").read())['encoding'], skip_header=1, delimiter="\t", filling_values = 0)
    lang='de'   
else:
    raise NotImplementedError(pm.language +" is not implemented yet!")

freqthreshold = 0.15 #1.5 #NV: why a threshold? For the french lexicon project, this reduces words from 38'000 to 1871. Therefore, almost no overlap
nr_highfreqwords = 500


def create_freq_file(freqlist_arrays, freqthreshold, nr_highfreqwords):
    
    #NV: every SUBTLEX or SUBTLEX equivalent has its own column names for the same thing, namely, the log of frequency of a word in that language
    
    ## Sort arrays ascending on subtlex by million
    if pm.language=='english':
        freqlist_arrays = np.sort(freqlist_arrays,order='LogFreqZipf')[::-1]
        select_by_freq = np.sum(freqlist_arrays['LogFreqZipf']>freqthreshold)
        freqlist_arrays = freqlist_arrays[0:select_by_freq]
        ## Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Spelling','LogFreqZipf']]

    elif pm.language=='french':
        freqlist_arrays = np.sort(freqlist_arrays,order='lcfreqmovies')[::-1]
        select_by_freq = np.sum(freqlist_arrays['lcfreqmovies']>freqthreshold)
        freqlist_arrays = freqlist_arrays[0:select_by_freq]        
        ## Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Word','lcfreqmovies']]


    elif pm.language=='german':
        freqlist_arrays = np.sort(freqlist_arrays,order='lgSubtlex')[::-1]
        select_by_freq = np.sum(freqlist_arrays['lgSubtlex']>freqthreshold)
        freqlist_arrays = freqlist_arrays[0:select_by_freq]
        ## Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Word','lgSubtlex']]

    
    frequency_words_np = np.empty([len(freq_words),1],dtype='U20')
    frequency_words_dict  = {}
    for i,line in enumerate(freq_words):
        frequency_words_dict[line[0].replace(".","").lower()] = line[1]
        frequency_words_np[i] = line[0].replace(".","").lower()
        
    cleaned_words = get_words(pm, task) #NV: merged get_words with get_words_task
    overlapping_words = np.intersect1d(cleaned_words,frequency_words_np, assume_unique=False) #NV: also removes duplicates


    print("words in task:\n",cleaned_words) #NV: uselful to check out if everything went well: see encoding of cleaned words, see percentage of overlap between dictionary and cleaned words
    print("amount of words in task:",len(cleaned_words))
    print("words in task AND in dictionnary:\n",overlapping_words)
    print("amount of overlapping words",len(overlapping_words))


    ## Match PSC/task and freq words and put in dictionary with freq
    file_freq_dict = {}
    for i,word in enumerate(overlapping_words):
        file_freq_dict[(word.lower()).strip()] = frequency_words_dict[word.strip()]

    ## Put top freq words in dict, can use np.shape(array)[0]):
    for line_number in range(nr_highfreqwords):
        file_freq_dict[((freq_words[line_number][0]).lower())] = freq_words[line_number][1]

    output_file_frequency_map = "Data/" + task + "_frequency_map_"+lang+".dat" #NV: input lang for every language. 
    with open (output_file_frequency_map,"wb") as f:
        pickle.dump(file_freq_dict,f)
    print('frequency file stored in '+output_file_frequency_map)
    return len(file_freq_dict)


def create_pred_file(task, file_freq_dict_length):
    file_pred_dict = np.repeat(0.25, file_freq_dict_length) #NV: why 539? Changed to file_freq_dict_length
    output_file_predictions_map = "Data/" + task + "_predictions_map_"+lang+".dat"
    with open (output_file_predictions_map,"wb") as f:
         pickle.dump(file_pred_dict,f)
    print('predictability file stored in '+output_file_predictions_map)

file_freq_dict_length=create_freq_file(freqlist_arrays,freqthreshold,nr_highfreqwords)
create_pred_file(task,file_freq_dict_length)
