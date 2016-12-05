import sys
import numpy as np
import theano
from theano import tensor as T
from generateForecastData import createForecastData
from neuralmodels.utils import permute
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import euclidean_loss
from neuralmodels.models import *
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import *
import os
import scipy.io as sio

if __name__ == '__main__':
    path = '/home/luna/ssp/srnn/dome/noisyRNN/checkpoints/Noise-FC500-ReLU-FC500-ReLU-LSTM1024-LSTM1024-FC500-ReLU-FC100-ReLU-FC42-ns5793-ep100-lrd0.97-wd0.001-noise7/'
    iteration = 5700
    path_to_checkpoint = '{0}checkpoint.{1}'.format(path, iteration)
    if os.path.exists(path_to_checkpoint):
        model = load(path_to_checkpoint)
        print 'Loaded model: ', path_to_checkpoint
        forecastDataDir = '/home/luna/ssp/srnn/data/single_original/160422_haggling11'
        [train_data, label_data] = createForecastData(forecastDataDir, 5, 500)
        print(train_data['s0'].shape)

        forecasted_motion1 = model.predict_sequence(train_data['s1'][:100, :, 3:], 500)
        forecasted_motion2 = model.predict_sequence(train_data['s2'][:100, :, 3:], 500)
        forecasted_motion0 = model.predict_sequence(train_data['s0'][:100, :, 3:], 500)

        predictionFilename = 'Haggling11ForecastMotion.mat'
        sio.savemat(os.path.join(path, predictionFilename),
                    {'s1_forecast': forecasted_motion1, 's2_forecast': forecasted_motion2,
                     's0_forecast': forecasted_motion0, 'gt': label_data})

        del model
    else:
        print 'Checkpoint path does not exist. Exiting!!'
        sys.exit()
