import tensorflow as tf
import os
import sys
import numpy as np
from six.moves import cPickle as pickle


IMG_SIZE = 32
NUM_CHANNELS = 3

def _read_pickle_from_file(filename):
   with tf.gfile.Open(filename, 'rb') as f:
       if sys.version_info >= (3, 0):
           data_dict = pickle.load(f, encoding='bytes')
       else:
           data_dict = pickle.load(f)
   return data_dict

def read_numpy(input_dir, file_names):
    train_data = None
    train_labels = None
    eval_data = None
    eval_labels = None
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        for input_file in input_files:
            data_dict = _read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            data = data.reshape(-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(0, 2, 3, 1)
            if mode is 'train':
                train_data = np.concatenate((train_data, data), axis=0) if train_data is not None else data
                train_labels = np.concatenate((train_labels, labels), axis=0) if train_labels is not None else labels
            elif mode is 'eval':
                eval_data = np.concatenate((eval_data, data), axis=0) if eval_data is not None else data
                eval_labels = np.concatenate((eval_labels, labels), axis=0) if eval_labels is not None else labels
    return train_data, train_labels, eval_data, eval_labels


def read_dataset(input_dir, file_names, batch_size):
    train_data, train_labels, eval_data, eval_labels = read_numpy(input_dir=input_dir, file_names=file_names)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    # Shuffle, repeat, and batch the examples.
    train_dataset = train_dataset.shuffle(1000).repeat().batch(batch_size)
    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_data, eval_labels))
    # Shuffle, repeat, and batch the examples.
    eval_dataset = eval_dataset.shuffle(1000).repeat().batch(batch_size)
    return train_dataset, eval_dataset

def read_train_batch(input_dir, file_names, batch_size):
    train_datsset, _ = read_dataset(input_dir=input_dir, file_names=file_names, batch_size=batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return next_element




