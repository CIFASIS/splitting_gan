import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir, shuffle=True, alt_labels=None):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    if alt_labels is not None:
        labels = np.array(alt_labels)
    else:
        labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    def get_epoch_unshuffled():
        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    if shuffle:
        return get_epoch
    else:
        return get_epoch_unshuffled


def load(batch_size, data_dir, **kwargs):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir, **kwargs),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )

