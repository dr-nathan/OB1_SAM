#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 1-10-2020 Noor Seijdel
# OB1 is a reading-model that simulates the processes behind reading in the brain.
# Here we simulate performance on experimental word-recognition tasks:
# a flanker task, a sentence reading task and an embeddedwords task

#from collections import defaultdict
#import re
from __future__ import division
import sys
import pickle

import numpy as np

from reading_common import stringToBigramsAndLocations, calcBigramExtInput, calcMonogramExtInput
from reading_functions import my_print, get_threshold, is_similar_word_length
# calc_saccade_error, norm_distribution, normalize_pred_values, middle_char, index_middle_char, \
# getMidwordPositionForSurroundingWord
from read_saccade_data import get_freq_pred_files, get_affix_file


def simulate_experiments(task, pm):

    if pm.visualise:  # NV: conditional import, so has to be imported after pm is specified
        import Visualise_reading

    lexicon = []
    lengtes = []
    all_data = []

    # NV: generate list of individual words and their lengths
    individual_words = []
    lengtes = []
    textsplitbyspace = list(pm.stim['all'].str.split(
        ' ', expand=True).stack().unique())  # get stimulus words of task
    for word in textsplitbyspace:
        if word.strip() != "":
            # For Python2 #NV: add _ to words, for affix recognition system
            new_word = f"_{word.strip().lower()}_"
            individual_words.append(new_word)
            lengtes.append(len(word))
    my_print(f'individual words: {individual_words}')

    # NV load appropriate dictionaries
    # get file of words of task for which their is a frequency and 200 most common words of language
    word_freq_dict_temp, word_pred_values = get_freq_pred_files(task, pm)

    # NV: also add _ to word_freq_dict, for affix modelling purposes.
    word_freq_dict = {}
    for word in word_freq_dict_temp.keys():
        word_freq_dict[f"_{word}_"] = word_freq_dict_temp[word]
    my_print('word freq dict (first 20): \n' +
             str({k: word_freq_dict[k] for k in list(word_freq_dict)[:20]}))

    # NV: also get data on frequency of affixes. NOTE: only works for english at the moment (prototype)
    affix_freq_dict_temp = get_affix_file(pm)
    affix_freq_dict = {}
    for word in affix_freq_dict_temp.keys():
        affix_freq_dict[f"{word}_"] = affix_freq_dict_temp[word] #NV: 

    affixes = list(affix_freq_dict.keys())
    my_print(affixes)

    # NV: add affix freq and pred to list
    (word_freq_dict, word_pred_values) = (word_freq_dict | affix_freq_dict,
                                          np.concatenate((word_pred_values, np.full(len(affix_freq_dict), 0.25))))

   # Replace prediction values with syntactic probabilities
    if pm.use_grammar_prob:
        print("grammar prob not implemented yet")
        raise NotImplementedError
        #word_pred_values = get_freq_and_syntax_pred()["pred"]
    if pm.uniform_pred:
        print("Replacing pred values with .25")
        # NS for the sentence experiment I will run this script twice, once with higher pred values (for the normal vs. scrambled conditions). Not the prettiest solution, but I would not know how to generate two different thresholds for the same word in the current framework.
        word_pred_values[:] = 0.25

    my_print(word_freq_dict)

    max_frequency_key = max(word_freq_dict, key=word_freq_dict.get)
    max_frequency = word_freq_dict[max_frequency_key]
    print("max freq:" + str(max_frequency))
    print("Length text: " + str(len(individual_words)) +
          "\nLength pred: " + str(len(word_pred_values)))
    # NV: uncommented, to make lists the same lengths
    # NV: (should already be the same length)
    word_pred_values = word_pred_values[0:len(individual_words)]

    # Make individual words dependent variables
    TOTAL_WORDS = len(individual_words)
    print("LENGTH of freq dict: "+str(len(word_freq_dict)))
    print("LENGTH of individual words: "+str(len(individual_words)))

    # make experiment lexicon (= dictionary + words in experiment)
    for word in individual_words:  # make sure it contains no double words
        if word not in lexicon:
            lexicon.append(word)

    if(len(word_freq_dict) > 0):
        for freq_word in word_freq_dict.keys():
            if freq_word.lower() not in lexicon:
                # NV:word_freq_dict already contains all target words of task +eventual flankers or primers, determined in create_freq_pred_files. So the first part of individual words is probably double work
                lexicon.append(freq_word.lower())

    lexicon_file_name = 'Data/Lexicon_'+task+'.dat'  # NV: Again, because word_freq_dict contained all words already, this is exactly the same file #ANSWER: Actually, the word_freq_dict is only made for words, for which there is a frequency. Other words are dicarderd. So concatenating word_freq_dict with individual_words puts those words back! So here again, the important question is: Why the threshols in create_freq_pred_files?
    with open(lexicon_file_name, "wb") as f:
        pickle.dump(lexicon, f)

    n_known_words = len(lexicon)  # nr of words known to model

    my_print(f'size lexicon: {len(lexicon)}')

    # Make lexicon dependent variables
    LEXICON_SIZE = len(lexicon)

    # Normalize word inhibition to the size of the lexicon.
    lexicon_normalized_word_inhibition = (100.0/LEXICON_SIZE) * pm.word_inhibition

    # Set activation of all words in lexicon to zero and make bigrams for each word.
    lexicon_word_activity = {}
    lexicon_word_bigrams = {}
    lexicon_word_bigrams_set = {}
    lexicon_index_dict = {}

    # Lexicon word measures
    # NV: print word activity ?? #TODO
    lexicon_word_activity_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_word_inhibition_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_word_inhibition_np2 = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_activewords_np = np.zeros((LEXICON_SIZE), dtype=int)
    word_input_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_thresholds_np = np.zeros((LEXICON_SIZE), dtype=float)

    word_thresh_dict = {}
    # for each word, compute threshold based on freq and pred
    # MM: dit is eigenlijk raar, threshold voor lex en voor indivwrds
    # NV: Mee eens. Daarbij zitten alle individual_words in het lexicon, dus is dit dubbel werk toch?
    for word in individual_words:
        word_thresh_dict[word] = get_threshold(word,
                                               word_freq_dict,
                                               max_frequency,
                                               pm.wordfreq_p,
                                               pm.max_threshold)
        try:
            word_freq_dict[word]
        except KeyError:
            word_freq_dict[word] = 0

    # list with trheshold values for words in lexicon
    for i, word in enumerate(lexicon):
        lexicon_thresholds_np[i] = get_threshold(word,
                                                 word_freq_dict,
                                                 max_frequency,
                                                 pm.wordfreq_p,
                                                 pm.max_threshold)
        lexicon_index_dict[word] = i
        lexicon_word_activity[word] = 0.0

    # lexicon indices for each word of text (individual_words)
    individual_to_lexicon_indices = np.zeros((len(individual_words)), dtype=int)
    for i, word in enumerate(individual_words):
        individual_to_lexicon_indices[i] = lexicon.index(word)

    # lexicon bigram dict
    N_ngrams_lexicon = []  # list with amount of ngrams per word in lexicon
    for word in lexicon:  # NV: was: range(LEXICON_SIZE):

        # NV: create local variable to modify without interfering
        word_local = word
        # determine if the word is an affix, and remove _'s temporarily
        is_affix = False
        if word_local.startswith('_'):
            word_local = word_local[1:-1]
        else:
            word_local = word_local[:-1]
            is_affix = True
        word_local = " "+word_local+" "
        (all_word_bigrams,
         bigramLocations) = stringToBigramsAndLocations(word_local, is_affix)  # NV: now returns special _ bigrams as well #again should be called suffix #NV: #TODO: should be made more elegant
        # lexicon[word] = lexicon[word][1:(len(lexicon[word]) - 1)]  # to get rid of spaces again

        lexicon_word_bigrams[word] = all_word_bigrams
        # append to list of N ngrams
        # bigrams and monograms total amount
        N_ngrams_lexicon.append(len(all_word_bigrams) + len(word))

    print("Amount of words in lexicon: ", LEXICON_SIZE)
    print("Amount of words in text:", TOTAL_WORDS)
    print("")

    # word-to-word inhibition matrix (redundant? we could also (re)compute it for every trial; only certain word combinations exist)
    # NV: could also be done once and pickled

    print("Setting up word-to-word inhibition grid...")
    # Set up the list of word inhibition pairs, with amount of bigram/monograms
    # overlaps for every pair. Initialize inhibition matrix with false.
    word_inhibition_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=bool)
    word_overlap_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=int)

    complete_selective_word_inhibition = True
    overlap_list = {}

    for other_word in range(LEXICON_SIZE):
        for word in range(LEXICON_SIZE):

            # NV: comment out temporarily, to investigate the effects of word-length-independent inhibition
            if False: #not is_similar_word_length(lexicon[word], lexicon[other_word]) or lexicon[word] == lexicon[other_word]: # Take word length into account here (instead of below, where act of lexicon words is determined)
                continue
            else:
                bigrams_common = []
                bigrams_append = bigrams_common.append
                bigram_overlap_counter = 0
                for bigram in range(len(lexicon_word_bigrams[lexicon[word]])):
                    if lexicon_word_bigrams[lexicon[word]][bigram] in lexicon_word_bigrams[lexicon[other_word]]:
                        bigrams_append(lexicon_word_bigrams[lexicon[word]][bigram])
                        lexicon_word_bigrams_set[lexicon[word]] = set(
                            lexicon_word_bigrams[lexicon[word]])
                        bigram_overlap_counter += 1

                monograms_common = []
                monograms_append = monograms_common.append
                monogram_overlap_counter = 0
                unique_word_letters = ''.join(set(lexicon[word]))

                for pos in range(len(unique_word_letters)):
                    monogram = unique_word_letters[pos]
                    if monogram in lexicon[other_word]:
                        monograms_append(monogram)
                        monogram_overlap_counter += 1

                # take into account both bigrams and monograms for inhibition counters (equally)
                total_overlap_counter = bigram_overlap_counter + monogram_overlap_counter

                # if word or other word is larger than the initial lexicon
                # (without PSC), overlap counter = 0, because words that are not
                # known should not inhibit
                if word >= n_known_words or other_word >= n_known_words:
                    total_overlap_counter = 0
                min_overlap = pm.min_overlap  # MM: currently 2

                if complete_selective_word_inhibition:  # NV: what does this do? #TODO
                    if total_overlap_counter > min_overlap:
                        word_overlap_matrix[word, other_word] = total_overlap_counter - min_overlap
                    else:
                        word_overlap_matrix[word, other_word] = 0
                else:  # is_similar_word_length
                    if total_overlap_counter > min_overlap:
                        word_inhibition_matrix[word, other_word] = True
                        word_inhibition_matrix[other_word, word] = True
                        overlap_list[word, other_word] = total_overlap_counter - min_overlap
                        overlap_list[other_word, word] = total_overlap_counter - min_overlap
                        sys.exit('Make sure to use slow version, fast/vectorized version not compatible')

    # Save overlap matrix, with individual words selected
    # NV: to automatically add right abbreviaton in file name
    output_inhibition_matrix = 'Data/Inhibition_matrix_'+pm.short[pm.language]+'.dat'
    with open(output_inhibition_matrix, "wb") as f:
        pickle.dump(np.sum(word_overlap_matrix, axis=0)[individual_to_lexicon_indices], f)
    print("Inhibition grid ready.")
    print("")
    print("BEGIN EXPERIMENT")
    print("")

    # Initialize Parameters
    # MM: voorste 3 kunnen weg toch?
    attendWidth = 8.0
    EyePosition = 0    #
    AttentionPosition = 0
    # milliseconds that one model cycle is supposed to last (brain time, not model time)
    CYCLE_SIZE = 25

    # allocated_dict = defaultdict(list)  # dictionary that will contain allocated words

    if pm.visualise:
        Visualise_reading

    # BEGIN EXPERIMENT
    # loop over trials
    stim = pm.stim
    stim['all'] = pm.stim['all']
    unrecognized_words = []  # NV: init empty list outside the for loop. Before, would only remember the last word

    for trial in range(0, len(stim['all'])):

        print("trial: " + str(trial+1))
        all_data.append({})

        stimulus = stim['all'][trial]

        stimulus_padded = " " + stimulus + " "

        # NV: eye position seems to be simply set in the beginning, and not manipulated (saccade blindness, etc)
        EyePosition = len(stimulus)//2
        # NV: in the case of embedded words, the eye position will be fixed to the center of the prime in the prime cycles, later.
        AttentionPosition = EyePosition

        all_data[trial] = {'stimulus': [],  # NV:#TODO: figure out why this exists
                           'prime': [],  # NV: added prime
                           'target': [],
                           'condition': [],
                           'cycle': [],
                           'lexicon activity per cycle': [],
                           'target activity per cycle': [],
                           'bigram activity per cycle': [],
                           'ngrams': [],
                           # 'recognized words indices': [],
                           # 'attentional width': attendWidth,
                           # 'exact recognized words positions': [],
                           'eye position': EyePosition,
                           'attention position': AttentionPosition,
                           'word threshold': 0,
                           'word frequency': 0,
                           'word predictability': 0,
                           'reaction time': [],
                           'correct': [],
                           'position': [],
                           'inhibition_value': pm.word_inhibition,  # NV: info for plots in notebook
                           'wordlen_threshold': pm.word_length_similarity_constant,
                           'error_rate': 0}  # NV: info for plots in notebook

        #my_print('attendWidth: '+str(attendWidth))

        shift = False

        # # Lexicon word measures
        lexicon_word_inhibition_np = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_total_input_np = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_word_activity_new = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_word_activity_np = np.zeros((LEXICON_SIZE), dtype=float)
        crt_word_activity_np = 0

        lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity

        if task == "Sentence":
            target = stimulus.split(" ")[stim['target'][trial]-1]  # read in target cue from file
            all_data[trial]['item_nr'] = stim['item_nr'][trial]
            all_data[trial]['position'] = stim['target'][trial]

        if task == "Flanker":
            #print("len stim: ", len(stimulus.split()))
            #print(len(stimulus.split()) // 2)
            if len(stimulus.split()) > 1:
                target = stimulus.split()[1]
            elif len(stimulus.split()) == 1:
                target = stimulus.split()[0]

        if task == "EmbeddedWords":  # NV added embedded words with values to be retained in all_data, and that are necessary later
            target = stim['target'][trial]
            prime = stim['prime'][trial]
            prime_padded = " " + prime + " "
            all_data[trial]['prime'] = prime
            all_data[trial]['item_nr'] = stim['item_nr'][trial]

        # store trial info in all_data
        all_data[trial]['stimulus'] = stimulus
        all_data[trial]['target'] = target

        # also add info on trial condition (read in from file? might be easiest)
        all_data[trial]['condition'] = stim['condition'][trial]

        # enter the cycle-loop that builds word activity with every cycle
        recognized = False
        cycle_for_RT = 0  # MM: used to compute RT
        cur_cycle = 0  # MM: current cycle (cycle counter)

        highest = None  # NV: reset highest activation index

        while cur_cycle < pm.totalcycles:  # NV: could be changed to a sequence of for loops. Or made abit more elegant via if curcycle in .. as described above

            # get allNgrams for current trial #NS added inside the loop to facilitate presentation of the stimulus in specific cycles

            # NV: dit is nu in het begin. IPV eerst bigrams ophalen en dan wissen, wordt nu gelijk de blank screen opgehaald

            # NV: during blank stimulus presentation at the beginning or at the end
            if cur_cycle < pm.blankscreen_cycles_begin or cur_cycle > pm.totalcycles-pm.blankscreen_cycles_end:

                if pm.blankscreen_type == 'blank':  # NV decide what type of blank screen to show
                    # NV: overwrite stimulus with empty string. Note: stimulus is not padded, but next function expects padded input, hence the name. (for the empty string it does not matter)
                    stimulus_padded = ""
                    # NV: Note: stimulus is not padded, but next function expects padded input, hence the name. (for the empty string it does not matter)
                    stimulus_padded = ""
                    my_print("Stimulus: blank screen")  # NV: show what is the actual stimulus

                elif pm.blankscreen_type == 'hashgrid':
                    stimulus = "#####"  # NV: overwrite stimulus with hash grid
                    stimulus_padded = " ##### "
                    my_print("Stimulus: hashgrid screen")  # NV: show what is the actual stimulus

            # NV: IF we are in priming cycle, set stimulus to the prime
            elif pm.is_priming_task and cur_cycle < (pm.blankscreen_cycles_begin+pm.ncyclesprime):
                stimulus = prime  # NV: overwrite stimulus with prime
                stimulus_padded = prime_padded
                my_print("Stimulus: "+stimulus)  # NV: show what is the actual stimulus

            else:
                # NV: reassign to change it back to original stimulus after prime or blankscreen.
                stimulus = stim['all'][trial]
                stimulus_padded = " "+stimulus+" "
                my_print("Stimulus: "+stimulus)

            # NV: else, just get the normal stimulus
            (allNgrams, bigramsToLocations) = stringToBigramsAndLocations(
                stimulus_padded, is_affix=False)
            EyePosition = len(stimulus)//2  # NV: After prime, set eye position back to stimulus
            AttentionPosition = EyePosition
            allMonograms = []
            allBigrams = []

            # NV: in elke cycle herbouw je de hele Ngram lijst, terwijl dat maar 1 keer hoeft (of 2 keer in priming task)
            for ngram in allNgrams:
                if len(ngram) == 2:
                    # NV: why split into bigrams and monograms? are never used
                    allBigrams.append(ngram)
                else:
                    allMonograms.append(ngram)
            allBigrams_set = set(allBigrams)
            my_print(allBigrams)

            unitActivations = {}  # reset after each trial

            # Reset
            word_input_np.fill(0.0)  # NV: reset word activity at each cycle?
            lexicon_word_inhibition_np.fill(0.0)
            lexicon_word_inhibition_np2.fill(0.0)
            lexicon_activewords_np.fill(False)

            # crt_trial_word_activities = [0] * len(stimulus.split()) #placeholder, different length for different stimuli (can be 1-5 words, depending on task and condition)

            # Calculate ngram activity
            # MM: could also be done above at start fix, and then again after attention shift. is constant in btw shifts
            for ngram in allNgrams:
                if len(ngram) == 2:
                    unitActivations[ngram] = calcBigramExtInput(ngram,
                                                                bigramsToLocations,
                                                                EyePosition,
                                                                AttentionPosition,
                                                                attendWidth,
                                                                shift,
                                                                cur_cycle)
                else:
                    unitActivations[ngram] = calcMonogramExtInput(ngram,
                                                                  bigramsToLocations,
                                                                  EyePosition,
                                                                  AttentionPosition,
                                                                  attendWidth,
                                                                  shift,
                                                                  cur_cycle)
                #
            all_data[trial]['bigram activity per cycle'].append(sum(unitActivations.values()))
            all_data[trial]['ngrams'].append(len(allNgrams))
            # activation of word nodes
            # taking nr of ngrams, word-to-word inhibition etc. into account
            wordBigramsInhibitionInput = 0
            for ngram in allNgrams:
                wordBigramsInhibitionInput += pm.bigram_to_word_inhibition * \
                    unitActivations[ngram]
            # This is where input is computed (excit is specific to word, inhib same for all)

            for lexicon_ix, lexicon_word in enumerate(lexicon):  # NS: why is this?
                wordExcitationInput = 0

                # (Fast) Bigram & Monogram activations
                bigram_intersect_list = allBigrams_set.intersection(
                    lexicon_word_bigrams[lexicon_word])
                for bigram in bigram_intersect_list:
                    wordExcitationInput += pm.bigram_to_word_excitation * \
                        unitActivations[bigram]
                for monogram in allMonograms:
                    if monogram in lexicon_word:
                        wordExcitationInput += pm.bigram_to_word_excitation * \
                            unitActivations[monogram]

                word_input_np[lexicon_ix] = wordExcitationInput + wordBigramsInhibitionInput

            # divide input by nr ngrams
            word_input_np = word_input_np / np.array(N_ngrams_lexicon)
            #print("word input divided by ngrams: " + str(word_input_np))

            # Active words selection vector (makes computations efficient)
            lexicon_activewords_np[(lexicon_word_activity_np > 0.0) | (word_input_np > 0.0)] = True

            # Calculate total inhibition for each word
            # Matrix * Vector (4x faster than vector)
            overlap_select = word_overlap_matrix[:, (lexicon_activewords_np == True)]
            lexicon_select = lexicon_word_activity_np[(lexicon_activewords_np == True)] * \
                lexicon_normalized_word_inhibition
            lexicon_word_inhibition_np = np.dot(overlap_select, lexicon_select)

            # Combine word inhibition and input, and update word activity
            lexicon_total_input_np = np.add(lexicon_word_inhibition_np, word_input_np)

            # now comes the formula for computing word activity.
            # pm.decay has a neg value, that's why it's here added, not subtracted
            # my_print("before:"+str(lexicon_word_activity_np[individual_to_lexicon_indices[fixation]]))
            lexicon_word_activity_new = ((pm.max_activity - lexicon_word_activity_np) * lexicon_total_input_np) + \
                                        ((lexicon_word_activity_np - pm.min_activity) * pm.decay)
            lexicon_word_activity_np = np.add(lexicon_word_activity_np, lexicon_word_activity_new)

            # Correct activity beyond minimum and maximum activity to min and max
            lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity
            lexicon_word_activity_np[lexicon_word_activity_np > pm.max_activity] = pm.max_activity

            # Save current word activities (per cycle)
            target_lexicon_index = individual_to_lexicon_indices[[
                idx for idx, element in enumerate(lexicon) if element == target]]
            my_print("target index:" + str(target_lexicon_index))

            #crt_word_total_input_np = lexicon_total_input_np[target_lexicon_index]
            crt_word_activity_np = lexicon_word_activity_np[target_lexicon_index]
            my_print("target activity:" + str(crt_word_activity_np))

            total_activity = 0

     #       for word in range(0,len(stimulus.split())): ## "stimulus activity" is now computed by adding the activations for each word (target and flankers)
     #           total_activity += lexicon_word_activity_np[lexicon_index_dict[stimulus.split()[word].lower()]] #TODO: NV: remove this, because tot activity is calculated on whole lexicon, as described below
            # MM: change tot act to act in all lexicon
            total_activity = sum(lexicon_word_activity_np)
            all_data[trial]['lexicon activity per cycle'].append(total_activity)
            print("total activity: "+str(total_activity))

            # Enter any recognized word to the 'recognized words indices' list
            # creates array (MM: msk?) that is 1 if act(word)>thres, 0 otherwise
            above_tresh_lexicon_np = np.where(
                lexicon_word_activity_np > lexicon_thresholds_np, 1, 0)
            all_data[trial]['cycle'].append(cur_cycle)
            all_data[trial]['target activity per cycle'].append(crt_word_activity_np)

            # array w. indices of recogn. words, not sure whether this still has a function
            # recognized_indices = np.asarray(all_data[trial]['recognized words indices'], dtype=int)
            my_print("nr. above thresh. in lexicon: " + str(np.sum(above_tresh_lexicon_np)))
            #my_print("recognized lexicon: ", above_tresh_lexicon_np)
            # NV: print words that are above threshold
            my_print("recognized words " +
                     str([x for i, x in enumerate(lexicon) if above_tresh_lexicon_np[i] == 1]))

            # NS: this final part of the loop is only for behavior (RT/errors)
            # array of zeros of len as lexicon, which will get 1 if wrd recognized
            new_recognized_words = np.zeros(LEXICON_SIZE)

            # but here we only look at the target word
            # MM: recognWrdsFittingLen_np: array with 1=wrd act above threshold, & approx same len
            # as to-be-recogn wrd (with 15% margin), 0=otherwise
            recognWrdsFittingLen_np = above_tresh_lexicon_np * \
                np.array([int(is_similar_word_length(x.replace('_', ''), target)) for x in lexicon]) #NV: replace('_', '') to remove underscores
                  
            # NV: #TODO: Build mechanism that splits word into stem + affix when affix is recognized

            # fast check whether there is at least one 1 in wrdsFittingLen_np
            if sum(recognWrdsFittingLen_np):
                # find word with the highest activation in all words that have a similar length
                highest = np.argmax(recognWrdsFittingLen_np * lexicon_word_activity_np)
                new_recognized_words[highest] = 1

                # NS if the target word is in recognized words:
                #print([lexicon[i] for i in np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0][:]])
                if target in [lexicon[i] for i in np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0][:]]:
                    #all_data[trial]['exact recognized words positions'].append(np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0][:])
                    recognized = True

            if recognized == False:
                cycle_for_RT = cur_cycle

            try:
                #print("target word: "+ target_word)
                my_print("highest activation of fitting length: " +
                         str(lexicon[highest])+", "+str(lexicon_word_activity_np[highest]))
                # print("\n")
            except:
                # NV: changed this, because the reason above print statement fails, is because there are no words above threshold
                my_print("no words above threshold and of fitting length")

            my_print("\n\n")  # NV: new lines for next cycle

            # NV: stop condition: if the design of the task calls for an end of trial after response, break. For now, a response is simply whenever a word gets above threshold
            if pm.trial_ends_on_key_press and sum(recognWrdsFittingLen_np) >= 1:
                break

            # NS: not yet implemented, potentially interesting for the future
            # "evaluate" response
                # e.g. through the Bayesian model Martijn mentioned (forgot to write it down),
                # or some hazard function that expresses the probability
                # of the one-choice decision process terminating in the
                # next instant of time, given that it has survived to that time?
            # if target word has been recognized (e.g. above threshold in time):
                ### response = word
                ### RT = moment in cylce
            # if target word has not been recognized:
                ### response = nonword
                ### RT = moment in cycle

            cur_cycle += 1

        # print("\n")
        if recognized == False:
            unrecognized_words.append(target)
            all_data[trial]['correct'].append(0)
        else:
            all_data[trial]['correct'].append(1)

        # MM: CHECK WHAT AVERAGE NON-DECISION TIME IS? OR RESPONSE EXECUTION TIME?
        reaction_time = (cycle_for_RT-(pm.blankscreen_cycles_begin +
                         pm.ncyclesprime)) * CYCLE_SIZE+300
        print("reaction time: " + str(reaction_time) + " ms")
        all_data[trial]['reaction time'].append(reaction_time)
        all_data[trial]['word threshold'] = word_thresh_dict.get(target, "")
        all_data[trial]['word frequency'] = word_freq_dict.get(target, "")
        # NV: added error info for plotting later. divide number of wrong words by total trials -> error rate. Added max(1,trial) to avoid divide by 0
        all_data[trial]['error_rate'] = str(len(unrecognized_words)/max(1, trial))

        print("end of trial")
        print("----------------")
        print("\n")

    # END OF EXPERIMENT. Return all data and a list of unrecognized words
    return lexicon, all_data, unrecognized_words
