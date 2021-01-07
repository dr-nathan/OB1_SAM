# -*- coding: utf-8 -*-
# 1-10-2020 Noor Seijdel
# In this file, "simulate_experiments" is called and the results are stored

from reading_simulation import reading_simulation
from simulate_experiments import simulate_experiments
from analyse_data_pandas import get_results
import multiprocessing as mp
import pickle
import cProfile
import pstats
from analyse_data_pandas import get_results, get_results_simulation
import pickle
import scipy
import time
import numpy as np
from get_scores import get_scores
import parameters_exp as pm
import pandas as pd
from get_parameters import get_params

# Get parameters for tuning
parameters, bounds, names = get_params(pm)

# Init distance variables for the reading function used in tuning
OLD_DISTANCE = np.inf
N_RUNS = 0

output_file_all_data, output_file_unrecognized_words = ("Results/all_data"+pm.language+".pkl","Results/unrecognized"+pm.language+".pkl")
start_time = time.time()

if pm.run_exp:
	# Run the reading model
	(lexicon, all_data, unrecognized_words) = simulate_experiments(parameters=[])
	# Save results: all_data...
	all_data_file = open(output_file_all_data,"w")
	pickle.dump(all_data, all_data_file)
	all_data_file.close()
	# ...and unrecognized words
	unrecognized_file = open(output_file_unrecognized_words, "w")
	pickle.dump(unrecognized_words, unrecognized_file)
	unrecognized_file.close()

	with open("unrecognized.txt", "w") as f:
                f.write("Total unrecognized: " + str(len(unrecognized_words)))
                f.write("\n")
                for uword in unrecognized_words:
                        f.write(str(uword))
                f.write("\n")

	with open("alldata" + str(it)+ ".txt", "w") as f:
                f.write("\n")
                for uword in all_data:
                        f.write(str(uword))
                f.write("\n")


if pm.analyze_results:
	get_results_simulation(output_file_all_data,output_file_unrecognized_words,it)


time_elapsed = time.time()-start_time
print("Time elapsed: "+str(time_elapsed))
