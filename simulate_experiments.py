#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 1-10-2020 Noor Seijdel
# OB1 is a reading-model that simulates the processes behind reading in the brain.
# Here we simulate performance on two experimental word-recognition tasks:
# a flanker task and a sentence reading task

from __future__ import division
from collections import defaultdict
import re
from reading_common import stringToBigramsAndLocations, calcBigramExtInput, calcMonogramExtInput, get_stimulus_text_from_file, calc_word_attention_right
from reading_functions import my_print, get_threshold, getMidwordPositionForSurroundingWord, is_similar_word_length, \
    calc_saccade_error, norm_distribution, normalize_pred_values, middle_char, index_middle_char
from read_saccade_data import get_freq_pred_files_fr, get_freq_and_syntax_pred
import numpy as np
import pickle
import parameters_exp as pm
import sys
#import create_freq_pred_files_fr
if pm.visualise:
    import Visualise_reading
import pandas as pd

def simulate_experiments(parameters):

    lexicon = []
    lengtes = []
    all_data = []

    # generate / read in stimuli list from file (fixed items for both experiments)
    if pm.use_sentence_task:
        #stim = pd.read_table('./Stimuli/Sentence_stimuli_all_csv.csv', sep=',')
        stim = pd.read_table('./Stimuli/debug_Sentence_stimuli_all_csv.csv', sep=',', encoding='utf-8')
        stim['all'] = stim['all'].astype('unicode')
        print(stim['all'])
        #stim['all'] = stim['all'].str.encode('utf-8')
        task = "Sentence"

    elif pm.use_flanker_task:
        stim = pd.read_table('./Stimuli/debug_Flanker_stimuli_all_csv.csv', sep=',')
        task = "Flanker"
        stim['all'] = stim['all'].astype('unicode')
        stim = stim[stim['condition'].str.startswith(('word'))].reset_index()

    #print(stim.head(10))
    individual_words = []
    lengtes=[]

    textsplitbyspace = list(stim['all'].str.split(' ', expand=True).stack().unique()) #stim['all'].str.split(expand=True).stack()#.reset_index(level = 0, name="all")
    #textsplitbyspace = set(stim['all'].str.lower().str.split(' ').sum())
    print(textsplitbyspace)
    for word in textsplitbyspace:
        if word.strip() != "":
            new_word = word.strip() #For Python2
            individual_words.append(new_word)
            lengtes.append(len(word))

    with open('./Texts/' + task + '_freq_pred.txt', 'w') as f:
        for word in individual_words:
            f.write('%s\n' % word.encode('utf-8'))

    print(individual_words)
    # load dictionaries (French Lexicon Project database) and generate list of individual words
    if pm.language == "french":
        word_freq_dict, word_pred_values = get_freq_pred_files_fr(task) # only freq?
    # Replace prediction values with syntactic probabilities
    if pm.use_grammar_prob:
        print("grammar prob not implemented yet")
        raise NotImplemented
            #word_pred_values = get_freq_and_syntax_pred()["pred"]
    if pm.uniform_pred:
        print("Replacing pred values with .25")
        word_pred_values[:] = 0.25

    print(word_freq_dict)

    max_frequency_key = max(word_freq_dict, key=word_freq_dict.get)
    max_frequency = word_freq_dict[max_frequency_key]
    print("Length text: " + str(len(individual_words)) + "\nLength pred: " + str(len(word_pred_values)))
    #word_pred_values = word_pred_values[0:len(individual_words)]

    # Make individual words dependent variables
    TOTAL_WORDS = len(individual_words)
    print("LENGTH of freq dict: "+str(len(word_freq_dict)))
    print("LENGTH of individual words: "+str(len(individual_words)))

    # make experiment lexicon (= dictionary + words in experiment)
    # make sure it contains no double words

    for word in individual_words:
        if word not in lexicon:
            lexicon.append(word)

    if(len(word_freq_dict)>0):
        for freq_word in word_freq_dict.keys():
            if freq_word.lower() not in lexicon:
                lexicon.append(freq_word.lower())

    lexicon_file_name = 'Data/Lexicon_fr.dat'
    with open(lexicon_file_name,"w") as f:
        pickle.dump(lexicon,f)

    n_known_words = len(lexicon)  # nr of words known to model
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
    lexicon_word_activity_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_word_inhibition_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_word_inhibition_np2 = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_activewords_np = np.zeros((LEXICON_SIZE), dtype=int)
    word_input_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_thresholds_np = np.zeros((LEXICON_SIZE), dtype=float)

    word_thresh_dict = {}
    # for each word, compute threshold based on freq and pred
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
    individual_to_lexicon_indices = np.zeros((len(individual_words)),dtype=int)
    for i, word in enumerate(individual_words):
        individual_to_lexicon_indices[i] = lexicon.index(word)

    # lexicon bigram dict
    N_ngrams_lexicon = []  # list with amount of ngrams per word in lexicon
    for word in range(LEXICON_SIZE):
        lexicon[word] = " "+lexicon[word]+" "
        [all_word_bigrams, bigramLocations] = stringToBigramsAndLocations(lexicon[word])
        lexicon[word] = lexicon[word][1:(len(lexicon[word]) - 1)]  # to get rid of spaces again
        lexicon_word_bigrams[lexicon[word]] = all_word_bigrams
        N_ngrams_lexicon.append(len(all_word_bigrams) + len(lexicon[word]))  # append to list of N ngrams

    print("Amount of words in lexicon: ", LEXICON_SIZE)
    print("Amount of words in text:", TOTAL_WORDS)
    print("")

    # word-to-word inhibition matrix (redundant? we could also (re)compute it for every trial; only certain word combinations exist)

    print ("Setting up word-to-word inhibition grid...")
    # Set up the list of word inhibition pairs, with amount of bigram/monograms
    # overlaps for every pair. Initialize inhibition matrix with false.
    word_inhibition_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=bool)
    word_overlap_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=int)

    complete_selective_word_inhibition = True
    overlap_list = {}

    for other_word in range(LEXICON_SIZE):
        for word in range(LEXICON_SIZE):
            # Take word length into account here (instead of below, where act of lexicon words is determinied)
            if not is_similar_word_length(lexicon[word], lexicon[other_word]) or lexicon[word] == lexicon[other_word]:
                continue
            else:
                bigrams_common = []
                bigrams_append = bigrams_common.append
                bigram_overlap_counter = 0
                for bigram in range(len(lexicon_word_bigrams[lexicon[word]])):
                    if lexicon_word_bigrams[lexicon[word]][bigram] in lexicon_word_bigrams[lexicon[other_word]]:
                        bigrams_append(lexicon_word_bigrams[lexicon[word]][bigram])
                        lexicon_word_bigrams_set[lexicon[word]] = set(lexicon_word_bigrams[lexicon[word]])
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

                if complete_selective_word_inhibition:
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
    output_inhibition_matrix = 'Data/Inhibition_matrix_fr.dat'
    with open(output_inhibition_matrix, "wb") as f:
        pickle.dump(np.sum(word_overlap_matrix, axis=0)[individual_to_lexicon_indices], f)
    print("Inhibition grid ready.")
    print("")
    print("BEGIN EXPERIMENT")
    print("")


    # Initialize Parameters
    regression = False
    wordskip = False
    refixation = False
    forward = False
    saccade_distance = 0  # Amount of characters
    fixation_duration = 0
    end_of_text = False  # Is set to true when end of text is reached.
    trial = 0
    trial_counter = 0  # The iterator that increases +1 with every trial,
    attendWidth = 8.0
    EyePosition = 0	#
    AttentionPosition = 0
    CYCLE_SIZE = 25  # milliseconds that one model cycle is supposed to last (brain time, not model time)

    allocated_dict = defaultdict(list)  # dictionary that will contain allocated words
    # defaultdict = dict that creates new entry each time that key does not yet exist.
    # (list): new entry will be empty list

    salience_position_new = pm.salience_position
    previous_lexicon_values = None
    reset_pred_previous = False
    N_in_allocated = 0
    N1_in_allocated = 0
    to_pauze = False


    if pm.visualise:
        Visualise_reading

    # BEGIN EXPERIMENT
    # loop over trials
    for trial in range(0,len(stim['all'])):
        print("trial: "+ str(trial))
        all_data.append({})

        stimulus = stim['all'][trial]
        print("stimulus: " + stimulus)

        stimulus_padded = " " + stimulus + " "

        #update EyePosition
        EyePosition = len(stimulus)//2
        print("eye position: " + str(EyePosition))

        AttentionPosition = EyePosition

        all_data[trial] = {'stimulus': [],
                            'target': [],
                            'condition': [],
                            'cycle': [],
                            'stimulus activity per cycle':[],
                            'target activity per cycle': [],
                            #'recognized words indices': [],
                            #'attentional width': attendWidth,
                            #'exact recognized words positions': [],
                            'eye position': EyePosition,
                            'attention position': AttentionPosition,
                            'word threshold': 0,
                            'word frequency': 0,
                            'word predictability': 0,
                            'reaction time': [],
                            'correct':[],
                            'position': []}

        my_print('attendWidth: '+str(attendWidth))

        shift = False

        lexicon_word_inhibition_np = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_total_input_np = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_word_activity_new = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_word_activity_np = np.zeros((LEXICON_SIZE), dtype=float)
        crt_word_activity_np = 0

        #print(len(stimulus.split(" ")))
        # # Lexicon word measures
        # lexicon_word_activity_np = np.zeros((LEXICON_SIZE), dtype=float)
        # lexicon_word_inhibition_np = np.zeros((LEXICON_SIZE), dtype=float)
        # lexicon_word_inhibition_np2 = np.zeros((LEXICON_SIZE), dtype=float)
        # lexicon_activewords_np = np.zeros((LEXICON_SIZE), dtype=int)
        # word_input_np = np.zeros((LEXICON_SIZE), dtype=float)
        # lexicon_thresholds_np = np.zeros((LEXICON_SIZE), dtype=float)

        lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity

        if pm.use_sentence_task:
            ncycles = 32#8 #8 cycles = 200 ms
            target = stimulus.split(" ")[stim['target'][trial]-1] #read in target cue from file
            all_data[trial]['position']=stim['target'][trial]
        if pm.use_flanker_task:
            ncycles = 32#6 #6 cycles = 150 ms
            print(stimulus.split())
            print("len stim: ", len(stimulus.split()))
            print(len(stimulus.split()) // 2)
            if len(stimulus.split())>1:
                target = stimulus.split()[1]
            elif len(stimulus.split())==1:
                target = stimulus.split()[0]

        print("target: "+ target)
        print("\n")

        # store trial info in all_data
        all_data[trial]['stimulus'] = stimulus
        all_data[trial]['target'] = target

        # also add info on trial condition (read in from file? might be easiest)
        all_data[trial]['condition'] = stim['condition'][trial]

        # everything below is still coded to potentially store/look at activations for all the words in the stimulus, and not only our target
        # for word in range(len(stimulus.split())):
        #     # "Word activities per cycle" is a dict containing the stimulus' words.
        #     # For every word there is a list that will keep track of the activity per cycle.
        #     all_data[trial]['word activities per cycle'].append(
        #         {stimulus.split()[word]: []}
        #     )

        crt_trial_word_activities_np = np.zeros((25,7),dtype=float)


        # # get allNgrams for current trial ## moved to inside the loop
        # [allNgrams, bigramsToLocations] = stringToBigramsAndLocations(stimulus_padded)
        # allMonograms = []
        # allBigrams = []
        #
        # for ngram in allNgrams:
        #     if len(ngram) == 2:
        #         allBigrams.append(ngram)
        #     else:
        #         allMonograms.append(ngram)
        # allBigrams_set = set(allBigrams)
        # allMonograms_set = set(allMonograms)

        # enter the cycle-loop that builds word activity with every cycle
        recognized = False
        amount_of_cycles = 0
        amount_of_cycles_before_end_of_trial = 0
            ### stimulus on screen for 150 ms (flanker) or 200 ms (sentence)

        while amount_of_cycles_before_end_of_trial < ncycles:
            # get allNgrams for current trial
            [allNgrams, bigramsToLocations] = stringToBigramsAndLocations(stimulus_padded)
            allMonograms = []
            allBigrams = []

            for ngram in allNgrams:
                if len(ngram) == 2:
                    allBigrams.append(ngram)
                else:
                    allMonograms.append(ngram)
            allBigrams_set = set(allBigrams)
            print(allBigrams)
            print(allBigrams_set)
            allMonograms_set = set(allMonograms)
            #print(allMonograms_set)
            if amount_of_cycles_before_end_of_trial < 8 or amount_of_cycles_before_end_of_trial > 16:
                [allNgrams, bigramsToLocations] = stringToBigramsAndLocations("")
                allMonograms = []
                allBigrams = []
                allBigrams_set = set(allBigrams)
                allMonograms_set = set(allMonograms)

            unitActivations = {}  # reset after each trials
            lexicon_activewords = []
            # Only the words in "lexicon_activewords" will later participate in word-to-word inhibition.
            # As such, less word overlap pairs will be called when calculating inhibition,
            # so to speed up the code.
            # Stores the indexes of the words in the lexicon are stored.

            # Reset
            word_input = []
            word_input_np.fill(0.0)
            lexicon_word_inhibition_np.fill(0.0)
            lexicon_word_inhibition_np2.fill(0.0)
            lexicon_activewords_np.fill(False)

            crt_trial_word_activities = [0] * len(stimulus.split()) #placeholder, different length for different stimuli (can be 1-5 words, depending on task and condition)

            ### Calculate ngram activity
            # MM: could also be done above at start fix, and then again after attention shift. is constant in btw shifts
            #print(allNgrams)
            for ngram in allNgrams:
                if len(ngram) == 2:
                    unitActivations[ngram] = calcBigramExtInput(ngram,
                                                                bigramsToLocations,
                                                                EyePosition,
                                                                AttentionPosition,
                                                                attendWidth,
                                                                shift,
                                                                amount_of_cycles_before_end_of_trial)
                else:
                    unitActivations[ngram] = calcMonogramExtInput(ngram,
                                                                  bigramsToLocations,
                                                                  EyePosition,
                                                                  AttentionPosition,
                                                                  attendWidth,
                                                                  shift,
                                                                  amount_of_cycles_before_end_of_trial)

            ### activation of word nodes
            # taking nr of ngrams, word-to-word inhibition etc. into account
            # MM: dit moet toch met een sum in 1 keer te doen zijn?
            wordBigramsInhibitionInput = 0
            for bigram in allBigrams:
                #print("bigram: "+ bigram)
                wordBigramsInhibitionInput += pm.bigram_to_word_inhibition * \
                                              unitActivations[bigram]
            for monogram in allMonograms:
                wordBigramsInhibitionInput += pm.bigram_to_word_inhibition * \
                                              unitActivations[monogram]

            # This is where input is computed (excit is specific to word, inhib same for all)
            for lexicon_ix, lexicon_word in enumerate(lexicon):
                wordExcitationInput = 0
                for ln in range(0, len(stimulus.split())):
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

            #print(wordExcitationInput)
            #print(wordBigramsInhibitionInput)
            #print(len(word_input_np))
            #print(len(N_ngrams_lexicon))
            #print(np.array(N_ngrams_lexicon))
            # MM: divide input by nr ngrams, because otherwise long wrds always a lot of input

            word_input_np = word_input_np / np.array(N_ngrams_lexicon) #NS
            #print("word input divided by ngrams: " + str(word_input_np))

            # Active words selection vector (makes computations efficient)
            lexicon_activewords_np[(lexicon_word_activity_np > 0.0) | (word_input_np > 0.0)] = True

            # Calculate total inhibition for each word
            # Matrix * Vector (4x faster than vector)
            overlap_select = word_overlap_matrix[:, (lexicon_activewords_np == True)]
            lexicon_select = lexicon_word_activity_np[(lexicon_activewords_np == True)] * \
                             lexicon_normalized_word_inhibition
            lexicon_word_inhibition_np = np.dot(overlap_select, lexicon_select)
            #print(lexicon_word_inhibition_np)

            # Combine word inhibition and input, and update word activity
            lexicon_total_input_np = np.add(lexicon_word_inhibition_np, word_input_np)

            # now comes the formula for computing word activity.
            # pm.decay has a neg value, that's why it's here added, not subtracted
            #my_print("before:"+str(lexicon_word_activity_np[individual_to_lexicon_indices[fixation]]))
            #print(lexicon_word_activity_np)
            lexicon_word_activity_new = ((pm.max_activity - lexicon_word_activity_np) * lexicon_total_input_np) + \
                                        ((lexicon_word_activity_np - pm.min_activity) * pm.decay)
            lexicon_word_activity_np = np.add(lexicon_word_activity_np, lexicon_word_activity_new)
            #print(lexicon_word_activity_np)

            #my_print("after:"+str( lexicon_word_activity_np[individual_to_lexicon_indices[fixation]]))
            # Correct activity beyond minimum and maximum activity to min and max
            lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity
            lexicon_word_activity_np[lexicon_word_activity_np > pm.max_activity] = pm.max_activity

            ## Save current word activities (per cycle)
            #print([idx for idx, element in enumerate(lexicon) if element == target_word])
            #print(target)
            target_lexicon_index = individual_to_lexicon_indices[[idx for idx, element in enumerate(lexicon) if element == target]]
            #print(target_lexicon_index)
            #print(target_lexicon_index[:])
            #print("Target_lexicon_index: ", target_lexicon_index)
            crt_word_total_input_np = lexicon_total_input_np[target_lexicon_index]
            crt_word_activity_np = lexicon_word_activity_np[target_lexicon_index]

            #print(crt_word_activity_np)
            # crt_trial_word_activities[2] = abs(lexicon_word_inhibition_np[target_lexicon_index])
            # print(crt_trial_word_activities)
            # crt_trial_word_activities_np[amount_of_cycles_before_end_of_trial, 2] = abs(lexicon_word_inhibition_np\
            #                                                                [target_lexicon_index])
            # crt_trial_word_activities_np[amount_of_cycles_before_end_of_trial, 5] = (pm.max_activity - crt_word_activity_np) * \
            #                                                         crt_word_total_input_np
            # crt_trial_word_activities_np[amount_of_cycles_before_end_of_trial, 6] = (crt_word_activity_np - pm.min_activity) * \
            #                                                         pm.decay

            total_activity = 0

            print("len: " + str(len(stimulus.split())))
            #print(lexicon_index_dict)
            #print(stimulus)
            if (len(stimulus.split())) > 0:
                for word in range(len(stimulus.split())):
                    total_activity += lexicon_word_activity_np[lexicon_index_dict[stimulus.split()[word]]]
            else:
                total_activity += lexicon_word_activity_np[lexicon_index_dict[stimulus.split()[0]]]

            all_data[trial]['stimulus activity per cycle'].append(total_activity)


			# Enter any recognized word to the 'recognized words indices' list
            # creates array that is 1 if act(word)>thres, 0 otherwise
            above_tresh_lexicon_np = np.where(lexicon_word_activity_np > lexicon_thresholds_np,1,0)
            msk = np.array([above_tresh_lexicon_np], dtype=bool)
            all_data[trial]['cycle'].append(amount_of_cycles_before_end_of_trial)
            all_data[trial]['target activity per cycle'].append(crt_word_activity_np)

            # array w. indices of recogn. words, not sure whether this still has a function
#            recognized_indices = np.asarray(all_data[trial]['recognized words indices'], dtype=int)
            #print(above_tresh_lexicon_np)
            my_print("above thresh. in lexicon: " + str(np.sum(above_tresh_lexicon_np)))
            my_print("recognized lexicon: ", above_tresh_lexicon_np)

            # array of zeros of len as lexicon, which will get 1 if wrd recognized
            new_recognized_words = np.zeros(LEXICON_SIZE)

            #but here we only look at the target word
            #for word_index in range(len(stimulus.split(" "))):
            desired_length = len(target)

            # MM: recognWrdsFittingLen_np: array with 1=wrd act above threshold, & approx same len
            # as to-be-recogn wrd (with 15% margin), 0=otherwise
            recognWrdsFittingLen_np = above_tresh_lexicon_np * np.array([int(is_similar_word_length(x, target)) for x in lexicon])

            # fast check whether there is at least one 1 in wrdsFittingLen_np
            if sum(recognWrdsFittingLen_np):
		    # find the word with the highest activation in all words that have a similar length
                highest = np.argmax(recognWrdsFittingLen_np * lexicon_word_activity_np)
                highest_word = lexicon[highest]
                new_recognized_words[highest] = 1

                #all_data[trial]['recognized words indices'].append(highest)
                # NS if the target word is in recognized words:
                #print([lexicon[i] for i in np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0][:]])
                if target in [lexicon[i] for i in np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0][:]]:
                    #all_data[trial]['exact recognized words positions'].append(np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0][:])
                    recognized = True

                    #print("target: " + target_word)

            if recognized == False:
                amount_of_cycles = amount_of_cycles_before_end_of_trial

            try:
                #print("target word: "+ target_word)
                print("highest activation: "+str(lexicon[highest])+", "+str(lexicon_word_activity_np[highest]))
                #print("\n")
            except:
                print("Encoding error")



            ### save activation for target word  for every cycle
            ## NS: not yet implemented, potentially interesting for the future
            ### "evaluate" response
                ## e.g. through the Bayesian model Martijn mentioned (forgot to write it down),
                ## or some hazard function that expresses the probability
                ## of the one-choice decision process terminating in the
                ## next instant of time, given that it has survived to that time?
            ### if target word has been recognized (e.g. above threshold in time):
                ### response = word
                ### RT = moment in cylce
            ### if target word has not been recognized:
                ### response = nonword
                ### RT = moment in cycle

            amount_of_cycles_before_end_of_trial += 1
        print("\n")
        unrecognized_words = []
        if recognized == False:
            unrecognized_words.append(target)
            all_data[trial]['correct'].append(0)
        else:
            all_data[trial]['correct'].append(1)

        reaction_time = (amount_of_cycles * CYCLE_SIZE)+300 #CHECK WHAT AVERAGE NON-DECISION TIME IS? OR RESPONSE EXECUTION TIME?
        print("reaction time: " + str(reaction_time) +" ms")
        all_data[trial]['reaction time'].append(reaction_time)
        all_data[trial]['word threshold'] = word_thresh_dict.get(target, "")
        all_data[trial]['word frequency'] = word_freq_dict.get(target,"")

        print("end of trial")
        print("----------------")
        print("\n")

    #print(all_data)
        # MM: Or implement ITIs to make residual act realistic? So simply loop with x time steps, and only decay..

    # END OF EXPERIMENT. Return all data and a list of unrecognized words
    return lexicon, all_data, unrecognized_words
