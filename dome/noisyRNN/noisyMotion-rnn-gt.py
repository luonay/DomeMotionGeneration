import numpy as np
import theano
from theano import tensor as T
from generateTrainValidDataonDomeData import createTrain
from neuralmodels.utils import permute
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import euclidean_loss 
from neuralmodels.models import *
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import *
import os
import scipy.io as sio

if __name__ == '__main__':
	#number of validation samples, each of length 1000. Only first /len_train_samples/ frames are fed into network 
	debug = 1

	num_valid_samples = 20 
	len_valid_samples = 500
	#total number of samples  = 22900
	num_train_samples = 300 - num_valid_samples
	len_train_samples =100
	data_shift = 50 #ajust shift size to generate more/less training samples

	epochs = 1
	batch_size = 100
	learning_rate = 1e-3
	learning_rate_decay = 0.97
	decay_after_ = 30
	weight_decay_ = 0.001
	decay_type_ = 'continuous'

	datadir = '/home/luna/ssp/data/single_original'
	
	[X_tr,Y_tr,X_valid_gt, Y_valid_gt]= createTrain(datadir,num_train_samples,len_train_samples, num_valid_samples, len_valid_samples, data_shift)
	if debug:
		print('size of raw input X_tr:{0}'.format(X_tr.shape))
		print('size of raw label Y_tr:{0}'.format(Y_tr.shape))
		print('size of raw valid input X_valid_gt:{0}'.format(X_valid_gt.shape))
		print('size of raw valid label Y_valid_gt:{0}'.format(Y_valid_gt.shape))
	#update true num_train_samples
	num_train_samples = X_tr.shape[1]
	num_valid_samples = X_valid_gt.shape[1]
	#num_validation = 50
	#num_train = num_samples - num_validation
	max_iter = X_tr.shape[1]/batch_size * epochs
	iter_per_epoch = X_tr.shape[1]/batch_size

        noise_rate_schedule_ = [0.0003, 0.0017, 0.003, 0.006, 0.01, 0.016, 0.023]
        noise_schedule_ = [500, 1000, 2000, 2600, 4000, 5000, 6600]

	permutation_train = permute(num_train_samples)
	#eliminate root location while training
	X_tr = X_tr[:,permutation_train,3:]
	Y_tr = Y_tr[:,permutation_train,3:]
	#X_tr = X[:,:num_train,:]
	#Y_tr = Y[:,:num_train,:]
	permutation_valid = permute(num_valid_samples)
	X_valid_gt = X_valid_gt[:,permutation_valid,:]
	Y_valid_gt = X_valid_gt[:,permutation_valid,:]
	#only the first len_train_samples frames are fed to the network. Tail parts are for comparison and visualization
	X_valid = X_valid_gt[:len_train_samples,:,3:]
	Y_valid = Y_valid_gt[:len_train_samples,:,3:]
	
	# Creating network layers
	l0 = TemporalInputFeatures(size=42)
	l1 = AddNoiseToInput()
        l2 = FCLayer(activation_str='identity', size=1024)
	l3 = LSTM(truncate_gradient=100,size=1024,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-5.0,g_high=5.0)
	l4 = LSTM(truncate_gradient=100,size=1024,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-5.0,g_high=5.0)
	l5 = LSTM(truncate_gradient=100,size=1024,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-5.0,g_high=5.0)
        l6 = FCLayer(activation_str='identity', size=42)
	layers = [l0, l1, l2, l3, l4, l5, l6]
	trY = T.tensor3(dtype=theano.config.floatX)
	lr = T.scalar()

		# Initializing network
	noisyRnn = noisyRNN(layers,euclidean_loss,trY,lr, weight_decay=weight_decay_)
	if debug:
		print('size of input X_tr:{0}'.format(X_tr.shape))
		print('size of input Y_tr:{0}'.format(Y_tr.shape))
	
        net_struct = 'Noise-FC1024-LSTM1024-LSTM1024-LSTM1024-FC42'
        hyper = '-ns' + str(num_train_samples) + '-ep' + str(epochs) + '-lrd' + str(learning_rate_decay) + '-wd' + str(weight_decay_) + '-noise' + str(len(noise_rate_schedule_))
        checkpointDir = '/home/luna/ssp/srnn/dome/noisyRNN/checkpoints/' + net_struct + hyper + '/'
	if not os.path.exists(checkpointDir):
		os.mkdir(checkpointDir)

	noisyRnn.fitModel(X_tr, Y_tr, 10*iter_per_epoch, checkpointDir, epochs, batch_size, learning_rate, learning_rate_decay, std=noise_rate_schedule_[0], decay_after=decay_after_, trX_validation=X_valid, trY_validation=Y_valid, decay_type=decay_type_, use_noise=True, noise_schedule=noise_schedule_, noise_rate_schedule=noise_rate_schedule_)

	#predict sequence of length (len_valid_samples - len_train_samples)
	out = noisyRnn.predict_sequence(X_valid,len_valid_samples - len_train_samples)
	sequence_predict = np.concatenate((X_valid, out), axis=0)
	if debug:
		print "predicted sequence shape:{0}".format(sequence_predict.shape)
	predictionFilename = 'genNoiseMotionwithGT.mat'
	sio.savemat(os.path.join(checkpointDir, predictionFilename), {'sequence_predict':sequence_predict, 'ground_truth':X_valid_gt})

	plot_loss(os.path.join(checkpointDir, 'logfile'),'train')
	plot_loss(os.path.join(checkpointDir, 'logfile'),'test')
