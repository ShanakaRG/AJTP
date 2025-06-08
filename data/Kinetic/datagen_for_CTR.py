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
    
    # m = 'sklearn'  # 'sklearn' or 'numpy'
    # Select validation set from training set
    # train_indices, val_indices = split_train_val(train_indices, m)

    # Save labels and num_frames for each sequence of each data set

    # Read the .npy file
    train_x= torch.tensor(np.load('./data/kinetics/train_data_joint.npy', mmap_mode='r'))

    N,C,T,V,M = train_x.shape
    train_x = train_x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C*V*M).numpy()

    test_x= torch.tensor(np.load('./data/kinetics/val_data_joint.npy', mmap_mode='r'))
    N,C,T,V,M = test_x.shape
    test_x = test_x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C*V*M).numpy()

    
    
    # Read the .pkl file
    with open('./data/kinetics/train_label.pkl', 'rb') as file:
        train_labels = pickle.load(file)

    with open('./data/kinetics/val_label.pkl', 'rb') as file:
        test_labels = pickle.load(file)

    print(type(train_labels))
    print(len(train_labels))
    print(len(train_labels[1]))
    train_y = one_hot_vector(train_labels[1])
    test_y = one_hot_vector(test_labels[1])

    
    
    # print(train_y)
    # train_x = skes_joints[train_indices]
    
    # test_x = skes_joints[test_indices]
    

    save_name = './data/kinetics/kinematics_400.npz' 
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

if __name__ == '__main__':
    split_dataset()