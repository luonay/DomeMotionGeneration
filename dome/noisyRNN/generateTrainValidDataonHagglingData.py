import numpy as np
import os

'''
prepare data for training from csv files.
each csv file contain N x 45 parentToChildVect for one subject
data prepared as list of length N, with each element a T x 42 matrix
T = len_sample, N = num_sample

returns a N x T x 45 ndarray
'''
def sample_data(input_data, len_samples, data_shift):
    overall_data = []
    i = 0
    while i * data_shift + len_samples < input_data.shape[0]:
        s_t = int(i * data_shift)
        e_t = int(i * data_shift + len_samples + 1)
        sample_data = input_data[s_t:e_t, 3:]
        overall_data.append(sample_data)
        i += 1
    overall_data = np.array(overall_data,dtype=np.float32)
    print overall_data.shape
    return overall_data


def createTrain(datadir, num_train_samples, len_train_samples, data_shift):
    overall_train_list = []
    train_len = 0
    for scene in os.listdir(datadir):
        if train_len >= num_train_samples:
            break
        if 'haggling' not in scene:
            continue
        train_scene = createTrainScene(os.path.join(datadir, scene), len_train_samples, data_shift)
        if type(train_scene).__name__ == 'int':
            continue
        train_len += train_scene.shape[0]
        overall_train_list.append(train_scene)
    overall_train_data = overall_train_list[0]
    s = 1
    while s < len(overall_train_list):
        overall_train_data = np.concatenate((overall_train_data, overall_train_list[s]), axis=0)
        s += 1
    overall_train_data = np.swapaxes(overall_train_data, 0, 1)
    if train_len > num_train_samples:
        overall_train_data = overall_train_data[:,:num_train_samples,:]
    print 'data: {0} distinct samples in total'.format(overall_train_data.shape[1])
    train_data = overall_train_data[:-1, :, :]
    train_label = overall_train_data[1:, :, :]
    #print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)
    return train_data, train_label

'''Each scene has three subjects. The function calls sample data for each subject and get a N x T x 45 tensor.
Then it concatenates the 3 tensors along axis=2 and returns a N x T x 135 tensor'''
def createTrainScene(scenefoldername, len_samples, data_shift):
    subjects = os.listdir(scenefoldername)
    print('{0} has {1} training subjects'.format(scenefoldername, len(subjects)))
    assert len(subjects) == 3
    sub_data = []
    for sub in subjects:
        input_data = np.loadtxt(os.path.join(scenefoldername, sub), delimiter=',')
        if input_data.shape[0] < len_samples:
            return 0
        sub_data.append(sample_data(input_data, len_samples, data_shift))
    overall_data = np.concatenate((sub_data[0], sub_data[1], sub_data[2]), axis=2)
    return overall_data


if __name__ == "__main__":
    datadir = '/home/luna/ssp/data/single_original'
    [train_data, train_label] = createTrain(datadir, 250, 100, 50)
    print(train_data.shape)
    print(train_label.shape)
    #print(valid_data.shape)
    #print(valid_label.shape)
    print(train_data[2, 1, 1:10] - train_label[1, 1, 1:10])
    print(train_data[24, 1, 1:10] - train_label[23, 1, 1:10])
    print(np.linalg.norm(train_data[3, 100, 3:6]))
    print(np.linalg.norm(train_data[24, 200, 6:9]))
