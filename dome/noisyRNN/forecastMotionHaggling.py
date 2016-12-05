import sys
import numpy as np
import theano
from theano import tensor as T
from generateForecastDataHaggling import createForecastData
from neuralmodels.utils import permute
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import euclidean_loss
from neuralmodels.models import *
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import *
import os
import scipy.io as sio

if __name__ == '__main__':
    path = '/home/luna/ssp/srnn/dome/noisyRNN/checkpoints/Noise-FC1024-LSTM1024-LSTM1024-LSTM1024-FC126-ns1995-ep70-lrd0.97-wd0.001-noise7/'
    iteration = 560
    path_to_checkpoint = '{0}checkpoint.{1}'.format(path, iteration)
    if os.path.exists(path_to_checkpoint):
        model = load(path_to_checkpoint)
        print 'Loaded model: ', path_to_checkpoint
        forecastDataDir = '/home/luna/ssp/srnn/data/single_original/160422_haggling12'
        [train_data, label_data] = createForecastData(forecastDataDir, 5, 500)
        print(train_data['s3'].shape)
        seed = np.concatenate((train_data['s3'][:100,:,3:], train_data['s4'][:100,:,3:], train_data['s5'][:100, :, 3:]), axis=2)

        forecasted_motion = model.predict_sequence(seed, 500)

        predictionFilename = 'Haggling12ForecastMotion.mat'
        sio.savemat(os.path.join(path, predictionFilename), {'forecast': forecasted_motion, 'gt': label_data})
        del model
    else:
        print 'Checkpoint path does not exist. Exiting!!'
        sys.exit()
