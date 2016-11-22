import numpy as np
import os

datadir = '/home/luna/ssp/data/single_original'
for scene in os.listdir(datadir):
	subjects = os.listdir(os.path.join(datadir, scene))
	for sub in subjects:
		filename = os.path.join(datadir, scene, sub)
		data = np.loadtxt(filename, delimiter=',')
		data = data[:,3:]
		if np.isnan(data).sum() > 0:
			print '{0} shape: {1}'.format(filename, data.shape)
			print 'mean: {0}, max: {1}, min: {2}, isnan: {3}'.format(np.mean(data), np.amax(data), np.amin(data), np.isnan(data).sum()) 
