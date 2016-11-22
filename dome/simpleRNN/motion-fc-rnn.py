import numpy as np
import theano
from theano import tensor as T
from generateTrainDataonDomeData import createTrain
from neuralmodels.utils import permute
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import euclidean_loss 
from neuralmodels.models import *
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import *
import os
import scipy.io as sio

if __name__ == '__main__':
	#total number of samples  = 22900
	num_samples = 22900 
	len_samples = 25

	epochs = 150
	batch_size = 100
	learning_rate_decay = 0.97
	decay_after=5

	datadir = '/home/luna/ssp/data/single_original'
	
	[X,Y]= createTrain(datadir,num_samples,len_samples)
	print('size of raw input X:{0}'.format(X.shape))
	print('size of raw label Y:{0}'.format(Y.shape))
	num_samples = X.shape[1]
	num_validation = 50
	num_train = num_samples - num_validation

	permutation = permute(num_samples)
	X = X[:,permutation,:]
	Y = Y[:,permutation,:]
	X_tr = X[:,:num_train,:]
	Y_tr = Y[:,:num_train,:]
	X_valid = X[:,num_train:,:]
	Y_valid = Y[:,num_train:,:]
	
	# Creating network layers
	l0 = TemporalInputFeatures(size=42)
	l1 = FCLayer(activation_str='rectify', size=500)
	l2 = FCLayer(activation_str='rectify', size=500) 
	l3 = LSTM(truncate_gradient=100,size=1024,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-10.0,g_high=10.0)
	l4 = LSTM(truncate_gradient=100,size=1024,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-10.0,g_high=10.0)
	l5 = FCLayer(activation_str='rectify', size=500) 
	l6 = FCLayer(activation_str='rectify', size=100) 
	l7 = FCLayer(activation_str='rectify', size=42) 
	layers = [l0, l1, l2, l3, l4, l5, l6, l7]

	trY = T.tensor3(dtype=theano.config.floatX)

	# Initializing network
	rnn = RNN(layers,euclidean_loss,trY,1e-3)

	net_struct = 'FC500-FC500-LSTM1024-LSTM1024-FC500-FC100-FC42-'
	hyper = 'ns' + str(num_samples) + '-ep' + str(epochs) + '-lrd' + str(learning_rate_decay) 
	checkpointDir = '/home/luna/ssp/srnn/dome/simpleRNN/checkpoints/' + net_struct + hyper + '/'
	if not os.path.exists(checkpointDir):
		os.mkdir(checkpointDir)

	print('size of input X_tr:{0}'.format(X_tr.shape))
	print('size of input Y_tr:{0}'.format(Y_tr.shape))
	# Fitting model
	rnn.fitModel(X_tr,Y_tr,X_valid, Y_valid,5,checkpointDir,epochs,batch_size,learning_rate_decay,decay_after)

	out = rnn.predict_motion_sequence(X_valid,1000)
	sequence_predict = np.concatenate((X_valid, out), axis=0)
	#out = out.squeeze(axis = (1,))
	print "predicted sequence shape:{0}".format(sequence_predict.shape)
	#print(out)
	predictionFilename = 'genMotion_trainsample.mat'
	sio.savemat(os.path.join(checkpointDir, predictionFilename), {'sequence_predict':sequence_predict})
	plot_loss(os.path.join(checkpointDir, 'logfile'))
