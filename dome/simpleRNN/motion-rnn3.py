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
	num_validation = 100
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
	l1 = LSTM(truncate_gradient=100,size=2048,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-10.0,g_high=10.0)
	l2 = LSTM(truncate_gradient=100,size=2048,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-10.0,g_high=10.0)
	l3 = LSTM(truncate_gradient=100,size=2048,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-10.0,g_high=10.0)
	l4 = LSTM(truncate_gradient=100,size=42,weights=None,seq_output=True,rng=None,
                skip_input=False,jump_up=False,grad_clip=True,g_low=-10.0,g_high=10.0)
	layers = [l0, l1, l2, l3, l4]
	#layers = [l0,l1, l3]
	#trY = T.dmatrix()
	trY = T.tensor3(dtype=theano.config.floatX)

	checkpointDir = '/home/luna/ssp/srnn/dome/simpleRNN/checkpoints/' + str(num_samples) + '_layer' + str(len(layers)-1) + '_epoch' + str(epochs) + '_size2048/'
	if not os.path.exists(checkpointDir):
		os.mkdir(checkpointDir)

	# Initializing network
	rnn = RNN(layers,euclidean_loss,trY,1e-3)

	print('size of input X_tr:{0}'.format(X_tr.shape))
	print('size of input Y_tr:{0}'.format(Y_tr.shape))
	# Fitting model
	rnn.fitModel(X_tr,Y_tr,X_valid, Y_valid,5,checkpointDir,epochs,batch_size,learning_rate_decay,decay_after)

	out = rnn.predict_motion_sequence(X_valid,1000)
	sequence_predict = np.concatenate((X_valid, out), axis=0)
	#out = out.squeeze(axis = (1,))
	print "predicted sequence shape:{0}".format(sequence_predict.shape)
	#print(out)
	predictionFilename = 'genMotion_trainsample' + str(num_samples) + '_layer' + str(len(layers)-1) + '_epoch' + str(epochs) + '.mat'
	sio.savemat(os.path.join(checkpointDir, predictionFilename), {'sequence_predict':sequence_predict})
	#np.savetxt(os.path.join(checkpointDir, 'generatedMotion.txt'), out, fmt='%f',delimiter=',')
	# Printing a generated sentence	
	#out = rnn.predict_language_model(X_valid[:,:1],1000,OutputSampleFromDiscrete)
	#print(out)
	
	# Print the sentence here
	#text_produced =  text_prediction(class_ids_reverse,out)
	#print(text_produced)
	plot_loss(os.path.join(checkpointDir, 'logfile'))
