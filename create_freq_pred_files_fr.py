import numpy as np
import pickle
import codecs
import parameters_exp as pm
#import chardet
from read_saccade_data import get_words, get_words_task, get_pred

# Detect text encoding
# rawdata=open("texts/frequency_german.txt","r").read()
# print chardet.detect(rawdata)

#to prevent unicode errors
# 'utf-8' 'cp1252' 'ISO-8859-1'
# with codecs.open("frequency_german.txt", 'r', encoding = 'ISO-8859-1',errors = 'strict') as mytext:

## Converter function to use
comma_to_dot = lambda s: float(s.replace(",","."))
decode_uft8 = lambda x: x.decode("utf-8", errors="strict")
decode_ISO= lambda x: x.decode('ISO-8859-1', errors="strict")
encode_uft8 = lambda x: x.encode("utf-8",errors="strict")
#replace_german = lambda x: unidecode(x)
to_lowercase = lambda x: x.lower()

## Set converters
convert_dict = {0:decode_ISO, 0:encode_uft8}
#{column:comma_to_dot for column in [4,5,9]}

## Get selected columns from text
freqlist_arrays = np.genfromtxt("Texts/frequency_french.txt", dtype=[('Word','U30'),('cfreqmovies','f4'), ('lcfreqmovies','f4'),('cfreqbooks','f4'), ('lcfreqbooks','f4')],
                                    usecols = (0,7,8,9,10), converters= convert_dict , skip_header=1, delimiter="\t", filling_values = 0)

freqthreshold = 1.5
nr_highfreqwords = 200

if pm.use_sentence_task:
    task = "Sentence"
elif pm.use_flanker_task:
    task = "Flanker"


def create_freq_file(freqlist_arrays, freqthreshold, nr_highfreqwords, task=task):
    ## Sort arrays ascending on frequency
    freqlist_arrays = np.sort(freqlist_arrays,order='lcfreqmovies')[::-1]
    select_by_freq = np.sum(freqlist_arrays['cfreqmovies']>freqthreshold)
    freqlist_arrays = freqlist_arrays[0:select_by_freq]

    ## Clean and select frequency words and frequency
    freq_words = freqlist_arrays[['Word','lcfreqmovies']]
    frequency_words_np = np.empty([len(freq_words),1],dtype='U20')
    frequency_words_dict  = {}
    for i,line in enumerate(freq_words):
        frequency_words_dict[line[0].replace(".","").lower()] = line[1]
        frequency_words_np[i] = line[0].replace(".","").lower()


    cleaned_words = get_words_task(task)
    overlapping_words = np.intersect1d(cleaned_words,frequency_words_np, assume_unique=False)

    ## IMPORTANT TO USE unicode() to place in dictionary, to replace NUMPY.UNICODE!!
    ## Match PSC and freq words and put in dictionary with freq
    file_freq_dict = {}
    for i,word in enumerate(overlapping_words):
        file_freq_dict[unicode(word.lower()).encode('utf-8').strip()] = frequency_words_dict[word.encode('utf-8').strip()]

    ## Put top freq words in dict, can use np.shape(array)[0]):
    for line_number in xrange(nr_highfreqwords):
        file_freq_dict[unicode((freq_words[line_number][0]).lower())] = freq_words[line_number][1]

    output_file_frequency_map = "Data/" + task + "_frequency_map_fr.dat"
    print(output_file_frequency_map)
    with open (output_file_frequency_map,"w") as f:
        pickle.dump(file_freq_dict,f)
        print("dumped")


def create_pred_file(task=task):
    #file_pred_dict = get_pred()
    file_pred_dict = np.repeat(0.25, 539)
    output_file_predictions_map = "Data/" + task + "_predictions_map_fr.dat"
    with open (output_file_predictions_map,"w") as f:
	    pickle.dump(file_pred_dict,f)

create_freq_file(freqlist_arrays,freqthreshold,nr_highfreqwords)
create_pred_file()
