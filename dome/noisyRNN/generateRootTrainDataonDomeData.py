import numpy as np 
import sys
import os
#prepare data for training from csv files.
#each csv file contain N x 45 parentToChildVect for one subject
#data prepared as list of length N, with each element a T x 42 matrix
#T = len_sample, N = num_sample
def sample_data(input_data,len_samples):
	overall_data = []
	train_data = []
	label_data = []
	#for i in range(num_samples):
	i = 0;
	while (i+1)*(len_samples+1) < input_data.shape[0]:
		#if (i+1)*(len_samples+1) >= input_data.shape[0]:
			#break
		s_t = int(i*len_samples)
		e_t = int((i+1)*len_samples + 1)
		sample_data = input_data[s_t:e_t, :]
		#overall_data.append([class_ids[x] fddor x in sample_text])
		overall_data.append(sample_data)
		i = i + 1
	#overall_data = np.array(overall_data,dtype=np.float64)
	#train_data = overall_data[:,:-1,3:]
	#label_data = overall_data[:,1:,3:]
	return overall_data

def createTrain(datadir,num_samples=1000,len_samples=25):
	overall_data = []
	for scene in os.listdir(datadir):
		if len(overall_data) > num_samples:
			break
		subjects = os.listdir(os.path.join(datadir,scene))
		print('scene {0} has {1} subjects'.format(scene, len(subjects)))
		for sub in subjects:
			filename = os.path.join(datadir, scene, sub)
			overall_data_sub = createTrainSubject(filename, len_samples)
			overall_data.extend(overall_data_sub)
			if (len(overall_data)>num_samples):
				overall_data = overall_data[:num_samples]
				break
	#print('training_data: {0} distinct samples in total'.format(len(overall_data)))
	overall_data = np.array(overall_data, dtype=np.float32)
	overall_data = np.swapaxes(overall_data, 0, 1)
	train_data = overall_data[:-1,:,:]
	label_data = overall_data[1:,:,:]
	return train_data, label_data

def createTrainSubject(filename, len_samples):
	input_data = np.loadtxt(filename, delimiter=',')
	#print '{1} frames in subject {0}'.format(os.path.basename(filename), input_data.shape[0])

	#[train_data,label_data] = sample_data(input_text,num_samples,len_samples,class_ids)
	overall_data = sample_data(input_data,len_samples)
	#train_data = np.swapaxis(train_data, 0, 1)
	#label_data = np.swapaxis(label_data, 0, 1)
	# dim = T x N x 45
	return overall_data

if __name__=="__main__":
	datadir = '/home/luna/ssp/data/single_original'
	[train_data, label_data] = createTrain(datadir,25000,25)
	print(train_data.shape)
	print(label_data.shape) 
	print(train_data[1,1,1:10] - label_data[1,0,1:10])
	print(train_data[100,1,1:10] - label_data[100,0,1:10])
	print(np.linalg.norm(train_data[100,5,3:6]))	
	print(np.linalg.norm(train_data[200,5,6:9]))	
	print(np.linalg.norm(train_data[300,8,9:12]))	
