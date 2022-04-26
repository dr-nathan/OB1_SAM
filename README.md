### Note April 2022: current state of model and recent mods
Thge inhibition matrix calculation is the most expensive step in the model. Therefore, the code now first checks if the last run was with the same parameters relevant for inhibition,
and if so, uses the previous inhibition matrix, thereby saving redundant computation.
In parallel, it also iterates through every word once instead of twice. 

The affix system is fully functional for english, which means there exists a pickle file containing affix frequencies, which can be used to anaylse affix-effects in simulations. However, frequency values are in the process of being updated, and added for french. The file affixes.py is in construction.

Turn affix_system in on in parameters.py for this functionality. 

For plotting inhibition and activation values during simulation, set plotting=True in parameters.py

### Note sept 2021 (MM)
This code based on Gina's handover version of April 2021, with Noor Seidel's code for simulating expts integrated in it. 
The code for experiments works well, that for reading text works but with hacks that are difficult to follow (must be rewritten).

# OB1 reader
OB1 is a reading-model that simulates the cognitive processes behind reading. 
For more information about the theory behind OB1 and how it works see: https://www.ncbi.nlm.nih.gov/pubmed/30080066

The code can be used for different purposes (code file mentioned are explained below). 

## Reading a text or running an experiment

In order to run a text reading or an experiment, one should set "run_exp" and "analyze_results" in *parameters.py* to True and "optimize" to False. 

### Text reading
To run the "normal" text reading task (Which means reading the input text-file once and comparing the results to an eye-tracking 
experiment), set task_to_run in *parameters.py* to "PSCall". In the standard version it reads a german text and uses word frequency as well as
word predictability (cloze probability) to recognize words presented in its visual field.
### Experiment 
To run an experiment, set task_to_run to the task in question. Can be "Flanker", from Snell et al (2019, Neuropsychologia), "Sentence", a reading experiment from Wen et al. 
(2019, Cognition), or "EmbeddedWords", a priming task from Beyersmann et al. (2016, Psychonomic Society).
The simulated experiment data is stored in pickled files called alldata_Flanker.pkl, etc. that can be read by Jupyter Notebooks written by Noor Seidel.
The Notebooks expect the pickled files in "...\Results". Run OB1_taskperformance before the other two (which compute ERPs and simulated ERPs, one for each task).
### NV: Update March 2022
in the latest version, a system for processing affixes has been implemented, to account for the priming results found by i.e. doi:10.3758/s13423-015-0927-z. In the present state, word pairs of the complex words and their stem (i.e. weaken - weak),
are detected and their inhibition is set to 0. That means that the word pairs dont inhibit each other, which explains why WEAKEN primes WEAK, but CASHEW does not prime CASH (no affix).
The affix system can be turned on or off by setting *affix_system* in parameters.py to True or False.
Additionally, the internal state of OB1 is recorded for every time step. This data can be consulted in the logfile.log

## Parameter-tuning 

In this version the model is executed multiple times in order to find the set of parameters that enables the model to 
read in a way that is similar to how a human would read the text. The optimization is done by using the *L-BFGS-B* 
optimization method from *scipy*.
For parameter-tuning define the parameters you wish to change and their bounds in *get_parameters.py*. Then go to 
*reading_simulation.py* where you have to unpack these values again based on the order in which they have been packed.
 
Next go to *parameters.py* and change "optimize" to True. Don't forget to set "run_exp" as well as "analyze_results" to
False if you want to **just optimize**.
  
The parameters are saved if they are better than the parameters from the previous iteration. They are saved
as a text file named after the tuning measure and the distance between experiment and simulation. 


## adding a new experiment 
When implementing a new task, head to parameters.py, input it in the list of possible tasks, and set its attributes. Add a CSV with stimuli in the /Stimuli map. 
Be careful to match the column structure of the other CSV's.

## adding a new language
To add a new language there has to be the plain text as input data for the reading simulation (file location defined in main.py, see *PSC_ALL.txt* as an example for the format), 
a lexicon (see word_freq.pkl as an example, file location defined in function "get_freq..." in read_saccade_data) as well as the preprocessed eyetracking-data recorded during an experiment
 where participants had to read the text that is presented to OB1. For an example of the input data derived from an eye-tracking experiment see the table stored in *Fixation_durations_german.pkl*.
The right files are now only available fo German/Potsdam corpus.

## files in the directory
The following files are the most important: 

### parameters.py
This is the most important function for controlling the behavior of *main.py*. Here the user can specify which parts of the programm should be run and also set the initial parameters when tuning. 
Furthermore the user can define which measures are used as error-function for the tuning process. 
This is also where the specifics of every task are specified.
**TODO:** Change to that locations text and lexicon are defined in parameters.

### main.py
In this file the main program flow is defined. In case of text reading it has calls to the reading_function, which simulates the actual reading, as imported from *reading_simulation.py*, the analyze function as imported from *analyse_data_pandas.py* and the optimize function, 
which is scipy's *L-BFGS-B* optimizing method. 
The function called by this optimizing method is a wrapper that takes the parameters called in *parameters.py* and feeds them to the reading simulation. The optimize function makes use of a slightly adapted version of the analyzing function that can be found in *get_scores.py*.
In case of experiment running, it calls simulate_experiments.

### reading_simulation.py
This file embodies the heart of the whole programm, the reading simulation. Here the input text is fed into the visuo-spatial representation which is activating bigramm-matrices, which in turn are activating words that are recognized. 
The resulting (correctly or incorrectly) recognized words are saved in **all_data**, together with a set of descriptive variables. 
At the end of the simulation this data-representation of the reading process is saved as a pickle file ( *all_data_INPUT_LANGUAGE.pkl* ) for analysis in a later stage together with all **unrecognized_words** ( *unrecognized_INPUT_LANGUAGE.pkl* ).
reading_simulation_BT.py is meant for the boundary task (BT). This sim has not been updated for a long while. If BT sims must be run, better use reading_simulation as basis and take whatever needed to do boundary from the BT file.
**TODO:** remake the code to the image of simulate_experiments.py

### simulate_experiments.py
This file has the code for simulating concrete experiments, currently a flanker expt from Snell et al (2019, Neuropsychologia), and a sentence reading experiment from Wen et al. 
(2019, Cognition), or an Embedded Words task (Beyersmann et al. 2016). In the flanker expt, a target word is presented either alone or surrounded by two flanker words on the screen. 
A hit is scored if the target word is recognized in timely fashion. In the sentence reading expt, a sentence (either correct or scrambled) is presented, and the reader has to read aloud a word indicated by a post cue. 
In the Embedded Words task, a prime is presented for 50ms, followed by a target. The user presses a button when the target word is recognized. The prime can be truly suffixed, pseudo suffixed or non-sufixed. head to Beyersamnn et al. (2020) for more info.
In the simulation, we count a hit when the cued word was recognized on time. In both experiments, total activity of word units is added up as substrate of the N400. This is then compared with experimental data using Jupyter Notebooks (see below). 

### reading_common.py
Helper-functions for the reading and experiment simulation 

### read_saccade_data.py
This file provides functions to read in the eye-tracking data collected during the experiment where participants had to read the same text that is presented to the OB1-reader. The functions for reading lexicons, word frequencies and cloze data are also here.

### analyse_data_plot.py / analyse_data_plot_qualitative.py
In this files the result of a single experiment is analyzed and plots as seen in the publication are produced.

### analyse_data_transformation / analyse_data_pandas.py
These files are providing various functions to analyze the data used in *analyse_data_pandas.py*
