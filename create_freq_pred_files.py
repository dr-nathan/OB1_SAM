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
import pandas as pd
import pickle
from read_saccade_data import get_words

pm = return_params()


# NV: This script only builds freq_pred file for task specified as task_to_run in parameters.py, and only for the language specified in parameters pm.language
task = pm.task_to_run

# NV: get appropriate freq dictionary (SUBTLEX-UK for english, Lexicon Project for french,...). Automatically detects encoding via Chardet and uses the value during import. Due to Chardet, its a bit slow however.
if pm.language == 'english':
    freqlist_arrays = pd.read_csv("Texts/SUBTLEX_UK.txt", usecols=(0, 1, 5), dtype={'Spelling': np.dtype(str)},
                                  encoding=chardet.detect(open("Texts/SUBTLEX_UK.txt", "rb").read())['encoding'], delimiter="\t")
    lang = 'en'
elif pm.language == 'french':
    freqlist_arrays = pd.read_csv("Texts/French_Lexicon_Project.txt",  usecols=(0, 7, 8, 9, 10), dtype={'Spelling': np.dtype(str)},
                                  encoding=chardet.detect(open("Texts/French_Lexicon_Project.txt", "rb").read())['encoding'], delimiter="\t")
    lang = 'fr'
elif pm.language == 'german':
    freqlist_arrays = pd.read_csv("Texts/SUBTLEX_DE.txt", usecols=(0, 1, 3, 4, 5, 9), dtype={'Spelling': np.dtype(str)},
                                  encoding=chardet.detect(open("Texts/SUBTLEX_DE.txt", "rb").read())['encoding'], delimiter="\t")
    lang = 'de'
else:
    raise NotImplementedError(pm.language + " is not implemented yet!")

freqthreshold = 0.15  # 1.5 #NV: why a threshold? For the french lexicon project, this reduces words from 38'000 to 1871. Therefore, almost no overlap
nr_highfreqwords = 500


def create_freq_file(freqlist_arrays, freqthreshold, nr_highfreqwords):

    # NV: every SUBTLEX or SUBTLEX equivalent has its own column names for the same thing, namely, the log of frequency of a word in that language

    # Sort arrays ascending on subtlex by million
    if pm.language == 'english':
        freqlist_arrays.sort_values(
            by=['LogFreq(Zipf)'], ascending=False,  inplace=True, ignore_index=True)
        # only keep above threshold words
        freqlist_arrays = freqlist_arrays[freqlist_arrays['LogFreq(Zipf)'] > freqthreshold]
        # Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Spelling', 'LogFreq(Zipf)']]
        freq_words.rename(columns={'Spelling': 'Word'}, inplace=True)

        # NV: is already in zipf, so no tranforming required

    elif pm.language == 'french':
        freqlist_arrays.sort_values(by=['cfreqmovies'], ascending=False,
                                    inplace=True, ignore_index=True)
        # only keep above threshold words #TODO: figure out filtering (right now, sorts on threshold, but scale is different for every language)
        # freqlist_arrays = freqlist_arrays[freqlist_arrays['cfreqmovies'] > freqthreshold]
        # Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Word', 'cfreqmovies']]

        # NV: convert to Zipf scale
        freq_words['LogFreq(Zipf)'] = freq_words['cfreqmovies'].apply(lambda x: np.log10(x*1000) if x>0 else 0) # from frequency per million to zipf. Also, replace -inf with 1

        freq_words.drop(columns=['cfreqmovies'], inplace=True)

    elif pm.language == 'german':

        freqlist_arrays.sort_values(by=['lgSubtlex'], ascending=False,
                                    inplace=True, ignore_index=True)
        # only keep above threshold words
        freqlist_arrays = freqlist_arrays[freqlist_arrays['lgSubtlex'] > freqthreshold]
        # Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Word', 'lgSubtlex']]

    frequency_words_np = np.empty([len(freq_words), 1], dtype='U20')
    frequency_words_dict = {}
    for ix, row in freq_words.iterrows():
        frequency_words_dict[row["Word"]] = row.iloc[1]  # get second column
        frequency_words_np[ix] = row.iloc[0]

    cleaned_words = get_words(pm, task)  # NV: merged get_words with get_words_task
    overlapping_words = np.intersect1d(
        cleaned_words, frequency_words_np, assume_unique=False)  # NV: also removes duplicates

    # NV: uselful to check out if everything went well: see encoding of cleaned words, see percentage of overlap between dictionary and cleaned words
    print("words in task:\n", cleaned_words)
    print("amount of words in task:", len(cleaned_words))
    print("words in task AND in dictionnary:\n", overlapping_words)
    print("amount of overlapping words", len(overlapping_words))

    # Match PSC/task and freq words and put in dictionary with freq
    file_freq_dict = {}
    for word in overlapping_words:
        file_freq_dict[(word.lower()).strip()] = frequency_words_dict[word.strip()]

    # Put top freq words in dict, can use np.shape(array)[0]):
    for line_number in range(nr_highfreqwords):
        file_freq_dict[((freq_words.iloc[line_number][0]).lower())
                       ] = freq_words.iloc[line_number][1]

    # NV: input lang for every language.
    output_file_frequency_map = "Data/" + task + "_frequency_map_"+lang+".dat"
    with open(output_file_frequency_map, "wb") as f:
        pickle.dump(file_freq_dict, f)
    print('frequency file stored in ' + output_file_frequency_map)

    # set global so it can be used for next function
    global file_freq_dict_length
    file_freq_dict_length = len(file_freq_dict)


def create_pred_file(task, file_freq_dict_length):
    # NV: why 539? Changed to file_freq_dict_length
    file_pred_dict = np.repeat(0.25, file_freq_dict_length)
    output_file_predictions_map = "Data/" + task + "_predictions_map_"+lang+".dat"
    with open(output_file_predictions_map, "wb") as f:
        pickle.dump(file_pred_dict, f)
    print('predictability file stored in '+output_file_predictions_map)


create_freq_file(freqlist_arrays, freqthreshold, nr_highfreqwords)
create_pred_file(task, file_freq_dict_length)
