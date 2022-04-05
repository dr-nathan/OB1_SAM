# -*- coding: UTF-8 -*-
# 1-10-2020 Noor Seijdel


# In this file, parameters are set and returned, task attributes are constructed and returned
import pandas as pd
import time
from types import SimpleNamespace


# global parameters that should be returned right away, task-independent
def return_global_params():
    """
    set parameters and task to run here
    possible tasks =  ["EmbeddedWords", "Sentence", "Flanker", "PSCall"] 
    NOTE: PSCall for normal text reading in german

    Returns
    -------
    object pm with parameters as attributes

    """

    task_to_run = 'EmbeddedWords'  # NV: task to run. can be Flanker, Sentence, EmbeddedWords or PSCall

    # NV: dictionnary for abbreviations, useful for filenames
    short = {'french': 'fr', 'german': 'de', 'english': 'en'}

    run_exp = True  # Should the experiment simulation run?
    analyze_results = False  # Should the results be analyzed?
    optimize = False  # Should the parameters be optimized using evolutionary algorithms?

    print_all = True
    plotting = False
    
    #for affix system
    simil_algo='lcs' #can be lev, lcs, startswith
    max_edit_dist=1 #NV: maximum allowed distance between word and inferred stem, to be considered matching (relates to affix system)
    short_word_cutoff=3

    return dict(locals())  # return dict of all local variables

# NV: Attributes of the relevant task, specified in global_params. In the form of object attributes. Allows all attributes to be bundled in one object. Also allows to set default values, which is useful when implementing new tasks
class TaskAttributes:

    def __init__(self, stim, stimAll, language, stimcycles, is_experiment,
                 is_priming_task=False, blankscreen_type='blank',
                 trial_ends_on_key_press=False, blankscreen_cycles_begin=0,
                 blankscreen_cycles_end=0, ncyclesprime=0):
        self.stim = stim
        self.stim['all'] = stimAll
        self.language = language
        self.stimcycles = stimcycles
        self.is_experiment = is_experiment
        self.is_priming_task = is_priming_task
        self.blankscreen_type = blankscreen_type
        self.trial_ends_on_key_press = trial_ends_on_key_press
        self.blankscreen_cycles_begin = blankscreen_cycles_begin
        self.blankscreen_cycles_end = blankscreen_cycles_end
        self.ncyclesprime = ncyclesprime
        self.totalcycles = self.blankscreen_cycles_begin + \
            self.ncyclesprime+self.stimcycles + self.blankscreen_cycles_end


# NV: When designing a new task, set its attributes here. csv must contain a column called 'all', which contains all elements that are on screen during target presentation
# NV: function returns instance of TaskAttributes with corresponding attributes
def return_attributes(task_to_run):
    if task_to_run == 'EmbeddedWords':
        stim = pd.read_csv('./Stimuli/EmbeddedWords_stimuli_all_csv.csv', sep=',')
        stim['all'] = stim['all'].astype('unicode')
        return TaskAttributes(
            stim,
            stim['all'],
            language='english',
            stimcycles=120,
            is_experiment=True,
            is_priming_task=True,
            blankscreen_type='hashgrid',
            trial_ends_on_key_press=True,
            blankscreen_cycles_begin=5,  # FIXME : 20
            blankscreen_cycles_end=0,
            ncyclesprime=2
        )
    elif task_to_run == 'Sentence':
        stim = pd.read_table('./Stimuli/Sentence_stimuli_all_csv.csv', sep=',')
        stim['all'] = stim['all'].astype('unicode')
        return TaskAttributes(
            stim,
            stim['all'],
            language='french',
            is_experiment=True,
            stimcycles=8,
            blankscreen_cycles_begin=8,
            blankscreen_cycles_end=16
        )
    elif task_to_run == 'Flanker':
        # NV: extra assignments needed for this task
        stim = pd.read_table('./Stimuli/Flanker_stimuli_all_csv.csv', sep=',')
        stim['all'] = stim['all'].astype('unicode')
        stim = stim[stim['condition'].str.startswith(('word'))].reset_index()
        return TaskAttributes(
            stim,
            stim['all'],
            language='french',
            stimcycles=6,
            is_experiment=True,
            blankscreen_cycles_begin=8,
            blankscreen_cycles_end=18
        )
    elif task_to_run == 'PSCall':  # TODO
        stim = None
        stim['all'] = None
        return TaskAttributes(
            stim,
            stim['all'],
            stimcycles=None,
            language='german',
            is_experiment=False)


# Control-flow parameters, that should be returned on a per-task basis
def return_task_params(task_attributes):

    # NV: if task is an experiment, choose this specific set of params (previously in parameters_exp.py)
    if task_attributes.is_experiment:

        use_grammar_prob = False  # True for using grammar probabilities, False for using cloze, overwritten by uniform_pred
        uniform_pred = True  # Overwrites cloze/grammar probabilities with 0.25 for all words

        include_sacc_type_sse = True  # Include the sse score based on the saccade type probability plot
        sacc_type_objective = "total"  # If "total" all subplots will be included in the final sse,
        #  single objectives can be "length", "freq" or "pred"

        include_sacc_dist_sse = True  # Include the SSE score derived from the saccade_distance.png plot

        tuning_measure = "SSE"  # can be "KL" or "SSE"
        discretization = "bin"  # can be "bin" or "kde"
        objective = []  # empty list for total SSE/KL, for single objectives: "total viewing time",
        # "Gaze durations", "Single fixations", "First fixation duration",
        # "Second fixation duration", "Regression"

        output_dir = time.time()
        epsilon = 0.1  # Step-size for approximation of the gradient

        ## Monoweight = 1
        decay = -0.05  # 0.08 #-0.053
        #inp. divided by #ngrams, so this param estimates excit per word [diff from paper]
        bigram_to_word_excitation = 4 # 1.25 
        # general inhibition on all words. The more active bigrams, the more general inhibition. #FIXME
        bigram_to_word_inhibition = -0.05 
        word_inhibition = -0.6  # -0.005 #-0.001  # -.0018 #-0.005#-0.07 #-0.0165
        # NV: determines how similar the length of 2 words must be for them to be recognised as 'similar word length'
        word_length_similarity_constant = 0.15

        letPerDeg = .3
        min_activity = 0.0
        max_activity = 1.3

        # Attentional width
        max_attend_width = 5.0
        min_attend_width = 3.0
        attention_skew = 4  # 1 equals symmetrical distribution # 4 (paper)
        bigram_gap = 2  # How many in btw letters still lead to bigram? 6 (optimal) # 3 (paper)
        min_overlap = 2
        refix_size = 0.2
        salience_position = 4.99  # 1.29 # 5 (optimal) # 1.29 (paper)
        corpora_repeats = 0  # how many times should corpus be repeated? (simulates diff. subjects)

        # Model settings
        frequency_flag = True  # use word freq in threshold
        prediction_flag = True
        similarity_based_recognition = True
        use_saccade_error = True
        use_attendposition_change = True  # attend width influenced by predictability next wrd
        visualise = False
        slow_word_activity = False
        pauze_allocation_errors = False
        use_boundary_task = False

        # Saccade error
        sacc_optimal_distance = 9.99  # 3.1 # 7.0 # 8.0 (optimal) # 7.0 (paper)
        saccErr_scaler = 0.2  # to determine avg error for distance difference
        saccErr_sigma = 0.17  # basic sigma
        saccErr_sigma_scaler = 0.06  # effect of distance on sigma

        # Fixation duration# s
        mu, sigma = 10.09, 5.36  # 4.9, 2.2 # 5.46258 (optimal), 4 # 4.9, 2.2 (paper)
        distribution_param = 5.0  # 1.1

        # Threshold parameters
        # MM: this is a HACK: a number of words have no freq because of a mistake, repaired by making freq less important
        max_threshold = 1  # 1
        # 0.4 # Max prop decrease in thresh for highest-freq wrd [different definition than in papers]
        wordfreq_p = 0.5  # 0.2 #NV: difference between max and min threshold
        wordpred_p = 0.2  # 0.4 # Currently not used

        task_params = dict(locals())
        # NV: task_attributes is given as input, so would end up in the namespace if not removed.
        task_params.pop('task_attributes')

        return task_params

    # NV: different set of parameters if task is not an experiment. (previously in parameters.py). This way, one can change params for experiments without interfering in PSC and vice versa
    else:

        use_grammar_prob = False  # True for using grammar probabilities, False for using cloze, overwritten by uniform_pred
        uniform_pred = False  # Overwrites cloze/grammar probabilities with 0.25 for all words

        include_sacc_type_sse = True  # Include the sse score based on the saccade type probability plot
        sacc_type_objective = "total"  # If "total" all subplots will be included in the final sse,
        #  single objectives can be "length", "freq" or "pred"

        include_sacc_dist_sse = True  # Include the SSE score derived from the saccade_distance.png plot

        tuning_measure = "SSE"  # can be "KL" or "SSE"
        discretization = "bin"  # can be "bin" or "kde"
        objective = []  # empty list for total SSE/KL, for single objectives: "total viewing time",
        # "Gaze durations", "Single fixations", "First fixation duration",
        # "Second fixation duration", "Regression"

        output_dir = time.time()
        epsilon = 0.1  # Step-size for approximation of the gradient

        ## Monoweight = 1
        decay = -0.08  # -0.053
        # 2.18 # inp. divded by #ngrams, so this param estimates excit per word [diff from paper]
        bigram_to_word_excitation = 3.09269333333
        bigram_to_word_inhibition = -0.20625  # -0.6583500000000001 # -0.55
        word_inhibition = -0.0165  # -0.016093 #-0.011 # -0.002
        word_length_similarity_constant = 0.35

        letPerDeg = .3
        min_activity = 0.0
        max_activity = 1.3

        # Attentional width
        max_attend_width = 5.0
        min_attend_width = 3.0
        attention_skew = 4  # 1 equals symmetrical distribution # 4 (paper)
        bigram_gap = 3  # How many in btw letters still lead to bigram? 6 (optimal) # 3 (paper)
        min_overlap = 2
        refix_size = 0.2
        salience_position = 4.99  # 1.29 # 5 (optimal) # 1.29 (paper)
        corpora_repeats = 0  # how many times should corpus be repeated? (simulates diff. subjects)

        # Model settings
        frequency_flag = True  # use word freq in threshold
        prediction_flag = True
        similarity_based_recognition = True
        use_saccade_error = True
        use_attendposition_change = True  # attend width influenced by predictability next wrd
        visualise = False
        slow_word_activity = False
        pauze_allocation_errors = False
        use_boundary_task = False

        # Saccade error
        sacc_optimal_distance = 9.99  # 3.1 # 7.0 # 8.0 (optimal) # 7.0 (paper)
        saccErr_scaler = 0.2  # to determine avg error for distance difference
        saccErr_sigma = 0.17  # basic sigma
        saccErr_sigma_scaler = 0.06  # effect of distance on sigma

        # Fixation duration# s
        mu, sigma = 10.09, 5.36  # 4.9, 2.2 # 5.46258 (optimal), 4 # 4.9, 2.2 (paper)
        distribution_param = 5.0  # 1.1

        # Threshold parameters
        max_threshold = 1
        # Max prop decrease in thresh. for highest-freq wrd [different definition than in papers]
        wordfreq_p = 0.4
        wordpred_p = 0.4  # Currently not used

        # Threshold parameters
        linear = False

        use_sentence_task = True
        use_flanker_task = False

        task_params = dict(locals())
        # NV: is given as input, so is a local variable. Would end up in the dictionary if not removed.
        task_params.pop('task_attributes')

        return task_params


def return_params():

    # NV: first, get global, task-independent variables. Values like which task to run, whether we want to run or optimize this task. returns dictionary
    global_params = return_global_params()

    # NV: fetch all attributes of the task to run, specified in parameters_exp. Creates an object
    task_attributes = return_attributes(global_params['task_to_run'])

    # NV: get parameters corresponding to type of given task (different set of parameters for experiment/non-experiment).returns dictionary
    task_params = return_task_params(task_attributes)

    # put all attributes of separate objects into one pm object
    pm = SimpleNamespace(**{**global_params, **task_attributes.__dict__, **task_params})

    return pm


if __name__ == '__main__':
    pm = return_params()
