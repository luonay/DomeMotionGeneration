import os
datadir = '/home/luna/ssp/data/single_original'
scene_list = []
for scene in os.listdir(datadir):
	#scene_list.append(scene)
	subjects = os.listdir(os.path.join(datadir,scene))
	print('scene {0} has {1} subjects'.format(scene, len(subjects)))
	for sub in subjects:
		filename = os.path.join(datadir, scene, sub)
		createTrainSubject(filename, len_samples)
	














