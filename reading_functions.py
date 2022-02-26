# CHANGED
__author__ = 'Sam van Leipsig'

import numpy as np

from parameters import return_params

pm=return_params()

## Basic
#---------------------------------------------------------------------------
def my_print(*args):
    if pm.print_all:
        for i in args:
            print(i)
        #print("")

#NV: the fucntion has been adapted to handle multipe lenghts values to be tested. Returns true if at least one length is matched, otherwise False
def is_similar_word_length(len1, lengths_to_be_matched):
    for len2 in lengths_to_be_matched:
        if abs(len1-len2) < (pm.word_length_similarity_constant * max(len1, len2)): # NV: difference of word length  must be within 15% of the length of the longest word
            return True
    return False


# returns the word center position of a surrounding word
# word position is > 0 for following words, < 0 for previous words
def getMidwordPositionForSurroundingWord(word_position,rightWordEdgeLetterIndexes,leftWordEdgeLetterIndexes):
    wordCenterPosition=None
    if word_position>0:
        word_slice_length = rightWordEdgeLetterIndexes[word_position][1]-rightWordEdgeLetterIndexes[word_position][0]+1
        wordCenterPosition = rightWordEdgeLetterIndexes[word_position][0]+round(word_slice_length/2.0)-1
    elif word_position==-1:
        previous_word_length = leftWordEdgeLetterIndexes[-2][1]-leftWordEdgeLetterIndexes[-2][0]+1
        wordCenterPosition = leftWordEdgeLetterIndexes[-2][0]+round(previous_word_length/2.0)-1
    return wordCenterPosition


## Reading
#---------------------------------------------------------------------------

#should always ensure that the maximum possible value of the threshold doesn't exceed the maximum allowable word activity
def get_threshold(word,word_freq_dict,max_frequency,freq_p,max_threshold, affixes):  #word_pred_dict,pred_p
    # let threshold be fun of word freq. freq_p weighs how strongly freq is (1=max, then thresh. 0 for most freq. word; <1 means less havy weighting)
    word_threshold = max_threshold # from 0-1, inverse of frequency, scaled to 0(highest freq)-1(lowest freq)
    if pm.frequency_flag:
        try:
            if word in affixes: #word in affixes:
                word_frequency = min(2*word_freq_dict[word], max_frequency) #NV: lower threshold for affixes (-> make freq higher, as it is linear), (but take max_frequency if freq is above max) 
            else:
                word_frequency = word_freq_dict[word]
                      
            word_threshold = max_threshold * ((max_frequency/freq_p) - word_frequency) / (max_frequency/freq_p) #threshold values between 0.8 and 1
        except KeyError:
            pass
        #GS Only lower threshold for short words
        #if len(word) < 4:
        #    word_threshold = word_threshold/3
        #return (word_frequency_multiplier * word_predictability_multiplier) * (pm.start_nonlin - (pm.nonlin_scaler*(math.exp(pm.wordlen_nonlin*len(word)))))
        
        return (word_threshold)#/1.4)


def normalize_pred_values(pred_p,pred_values):
    max_predictability = 1.
    return ((pred_p * max_predictability) - pred_values) / (pred_p * max_predictability)


# Make sure saccError doesn't cause NextEyeposition > stimulus
def calc_saccade_error(saccade_distance,optimal_distance,saccErr_scaler,saccErr_sigma,saccErr_sigma_scaler):
    #TODO include fixdur, as in EZ and McConkie (smaller sacc error after longer fixations)
    saccade_error = (optimal_distance - abs(saccade_distance)) * saccErr_scaler
    saccade_error_sigma = saccErr_sigma + (abs(saccade_distance) * saccErr_sigma_scaler)
    saccade_error_norm = np.random.normal(saccade_error,saccade_error_sigma,1)
    if pm.use_saccade_error:
        return saccade_error_norm
    else:
        return 0.


def norm_distribution(mu,sigma,distribution_param,recognized):
    if recognized:
        return int(np.round(np.random.normal(mu-distribution_param,sigma,1)))
    else:
        return int(np.round(np.random.normal(mu,sigma,1)))

def middle_char(txt):
   return txt[(len(txt)-1)//2:(len(txt)+2)//2]

def index_middle_char(txt):
    return ((len(txt))//2)
