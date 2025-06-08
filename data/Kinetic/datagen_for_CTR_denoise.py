import os
import os.path as osp
import numpy as np
import pickle
import logging
import h5py
from sklearn.model_selection import train_test_split
import pickle
import torch 
def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 400))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_dataset():

    # Save labels and num_frames for each sequence of each data set

    # Read the .npy file
    train_x= torch.tensor(np.load('train_data_joint.npy', mmap_mode='r'))

    frame_sums = torch.sum(train_x, dim=(1, 2, 3,4))

    # Find the indices of non-corrupted data samples
    non_corrupted_indices = torch.nonzero(frame_sums).squeeze()
    train_x = train_x[non_corrupted_indices]
    N,C,T,V,M = train_x.shape
    train_x = train_x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C*V*M).numpy()

    # Read the .pkl file
    with open('train_label.pkl', 'rb') as file:
         train_labels = pickle.load(file)
    train_labels = [ train_labels[1][i] for i in non_corrupted_indices]
    train_y = one_hot_vector(train_labels)
    
    test_x= torch.tensor(np.load('val_data_joint.npy', mmap_mode='r'))
    frame_sums = torch.sum(test_x, dim=(1, 2, 3,4))
    non_corrupted_indices = torch.nonzero(frame_sums).squeeze()
    test_x = test_x[non_corrupted_indices]
    N,C,T,V,M = test_x.shape
    test_x = test_x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C*V*M).numpy()

    with open('val_label.pkl', 'rb') as file:
        test_labels = pickle.load(file)

    test_labels = [ test_labels[1][i] for i in non_corrupted_indices]
    test_y = one_hot_vector(test_labels)

    save_name = 'kinematics_400.npz' 
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

if __name__ == '__main__':
    split_dataset()