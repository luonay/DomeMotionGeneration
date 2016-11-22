import numpy as np 
from collections import Counter
import sys
#prepare data for training from csv files.
#each csv file contain N x 45 parentToChildVect for one subject
#data prepared as list of length N, with each element a T x 42 matrix
#T = len_sample, N = num_sample
def sample_data(input_data,num_samples,len_samples):
	overall_data = []
	train_data = []
	label_data = []
	for i in range(num_samples):
		if (i+1)*(len_samples+1) >= input_data.shape[0]:
			break
		s_t = int(i*len_samples)
		e_t = int((i+1)*len_samples + 1)
		sample_data = input_data[s_t:e_t, :]
		#overall_data.append([class_ids[x] fddor x in sample_text])
		overall_data.append(sample_data)
	#overall_data = np.array(overall_data,dtype=np.int64)
	train_data = overall_data[:,:-1]
	label_data = overall_data[:,1:]
	return train_data,label_datg


