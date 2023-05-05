import torch
import torch.utils.data as data
import os
import numpy as np
import copy

class train_data_loader(data.Dataset):
    def __init__(self, data_npz_path, dataset_name):
        assert os.path.exists(data_npz_path) == True

        self.dataset_name = dataset_name

        vector_data = np.load(data_npz_path)['vectors']
        spker_label = np.load(data_npz_path)['spker_label']
        utt_label = np.load(data_npz_path)['utt_label']

        self.vector_data = vector_data  # vectors
        self.spker_label = spker_label  # spker label
        self.utt_label = utt_label  # utt label

        self.spker_class = np.unique(self.spker_label)
        self.spker_data = self.get_spker_data()

        print("dataset: {}".format(dataset_name))
        print("vectors shape: ", np.shape(vector_data))
        print("spker label shape: ", np.shape(spker_label))
        print("num of spker: ", np.shape(np.unique(spker_label)))
        print("utt label shape: ", np.shape(utt_label))
        print("num of utt: ", np.shape(np.unique(utt_label)))

    def get_spker_data(self):
        '''build a hash map spk->vecs'''
        spker_data = {}
        DATA_DIM = np.shape(self.data)[1]
        for i in range(len(self.data)):
            key = self.spker_label[i]
            if key not in spker_data.keys():
                spker_data[key] = np.reshape(self.data[i], (-1, DATA_DIM))
            else:
                spker_data[key] = np.vstack((spker_data[key], self.data[i]))
        return spker_data

    def __len__(self):
        return len(self.spker_class)

    def __getitem__(self, index):
        return self.spker_class[index]

    @property
    def data(self):
        return self.vector_data

    @property
    def label(self):
        return self.spker_label


class test_data_loader(data.Dataset):
    def __init__(self, data_npz_path, dataset_name):
        assert os.path.exists(data_npz_path) == True

        self.dataset_name = dataset_name

        vector_data = np.load(data_npz_path)['vectors']
        spker_label = np.load(data_npz_path)['spker_label']
        utt_label = np.load(data_npz_path)['utt_label']

        self.vector_data = vector_data  # vectors
        self.spker_label = spker_label  # spker label
        self.utt_label = utt_label  # utt label

        print("dataset: {}".format(dataset_name))
        print("vectors shape: ", np.shape(vector_data))
        print("spker label shape: ", np.shape(spker_label))
        print("num of spker: ", np.shape(np.unique(spker_label)))
        print("utt label shape: ", np.shape(utt_label))
        print("num of utt: ", np.shape(np.unique(utt_label)))

    def __len__(self):
        return len(self.vector_data)

    def __getitem__(self, index):
        return self.vector_data[index], self.spker_label[index]

    @property
    def data(self):
        return self.vector_data

    @property
    def label(self):
        return self.spker_label


if __name__ == "__main__":
    pass
