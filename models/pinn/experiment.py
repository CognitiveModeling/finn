"""
This script either trains n models or tests them as determined in the
config.json file.
"""

import multiprocessing as mp
import numpy as np
import train
import test

#
# Parameters

TRAIN = False
MODEL_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Number of models that can be trained in parallel
POOLSIZE = 1


#
# SCRIPT

if TRAIN:
	
	# Train the MODEL_NUMBERS using POOLSIZE threads in parallel
	if POOLSIZE > 1:
		arguments = list()
		for model_number in MODEL_NUMBERS:
			arguments.append([False, model_number])

		with mp.Pool(POOLSIZE) as pool:
			pool.starmap(train.run_training, arguments)
	else:
		for model_number in MODEL_NUMBERS:
			train.run_training(print_progress=False, model_number=model_number)

else:
	# Test the MODEL_NUMBERS iteratively
	error_list = list()
	for model_number in MODEL_NUMBERS:
		mse, pred, labels = test.run_testing(model_number=model_number)
		error_list.append(mse)

	print(error_list)
	print(f"Average MSE: {np.mean(error_list)}, STD: {np.std(error_list)}")
