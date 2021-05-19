# -*- coding: utf-8 -*-
"""
Created on Wed Apri 7 16:41:28 2021
@author: guemesturb
"""


import os
import re
import numpy as np
import scipy.io as sio
import tensorflow as tf
import scipy.signal
import cv2



def generate_gaussianfilter_training_pipeline(tfr_path, channels, n_modes, filter_size,order_noise = 0, validation_split=0.2, batch_size=8, shuffle_buffer=400, n_prefetch=4, cpu=False):
    # List all files in tfr_path folder

    tfr_files = sorted([os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])

    # Keep only files terminated on .tfrecords

    regex = re.compile(f'.tfrecords')
    tfr_files = ([string for string in tfr_files if re.search(regex, string)])

    # Separating files for training and validation
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))
    tot_samples_per_ds = sum(n_samples_per_tfr)
    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-2][-3:])
    tfr_files = [string for string in tfr_files if int(string.split('_')[-2][:3]) <= n_tfr_loaded_per_ds]
    n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))
    n_samp_valid = sum(n_samples_per_tfr) - n_samp_train
    (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

    if samples_train_left > 0:

        n_files_train += 1

    tfr_files_train = [string for string in tfr_files if int(string.split('_')[-2][:3]) <= n_files_train]
    n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1


    if sum([int(s.split('.')[-2][-3:]) for s in tfr_files_train]) != n_samp_train:
        shared_tfr = tfr_files_train[-1]
        tfr_files_valid = [shared_tfr]
    else:
        shared_tfr = ''
        tfr_files_valid = list()

    tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])
    tfr_files_valid = sorted(tfr_files_valid)
    shared_tfr_out = tf.constant(shared_tfr)
    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
    n_samples_loaded_per_tfr = list()

    if n_tfr_loaded_per_ds>1:

        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])

    else:

        n_samples_loaded_per_tfr.append(tot_samples_per_ds)

    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)
    tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)
    tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

    if n_tfr_left-1>0:

        samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left-2]
        n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left-1]

    else:

        samples_train_shared = samples_train_left
        n_samples_tfr_shared = n_samples_loaded_per_tfr[0]
 
    tfr_files_train_ds = tfr_files_train_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[3],sep='-')[0], tf.int32)-1)), 
        cycle_length=16, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    tfr_files_val_ds = tfr_files_val_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-2],sep='-')[0], tf.int32)-1)),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if cpu:

        dataset_train = tfr_files_train_ds.map(lambda x: tf_parser_training_cpu(x, tfr_path, channels, n_modes, filter_size, order_noise), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_valid = tfr_files_val_ds.map(lambda x: tf_parser_training_cpu(x, tfr_path, channels, n_modes, filter_size, order_noise), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    else:

        dataset_train = tfr_files_train_ds.map(lambda x: tf_parser_training(x, tfr_path, channels, n_modes, filter_size, order_noise), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_valid = tfr_files_val_ds.map(lambda x: tf_parser_training(x, tfr_path, channels, n_modes, filter_size, order_noise), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_train = dataset_train.shuffle(shuffle_buffer)
    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train = dataset_train.prefetch(n_prefetch)
    
    dataset_valid = dataset_valid.shuffle(shuffle_buffer)
    dataset_valid = dataset_valid.batch(batch_size=batch_size)
    dataset_valid = dataset_valid.prefetch(n_prefetch)

    return dataset_train, dataset_valid






def filtering(A,filter_size):
    channel = 3
    n_z = 64
    n_x = 128
    F_1 = cv2.getGaussianKernel(ksize=filter_size,sigma=1)
    kernel_in = F_1*np.transpose(F_1)
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    img  = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(A,axis = 0),axis = 3), kernel, strides=[1, 1, 1, 1], padding='SAME')
    return img[0,:,:,0]



@tf.function
def tf_parser_training(rec, tfr_path, channels, n_modes, filter_size,order_noise):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''
    features = {
        'i_samp': tf.io.FixedLenFeature([], tf.int64),
        'n_x': tf.io.FixedLenFeature([], tf.int64),
        'n_z': tf.io.FixedLenFeature([], tf.int64),
        'wall_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'psi': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }

    parsed_rec = tf.io.parse_single_example(rec, features)

    nx = tf.cast(parsed_rec['n_x'], tf.int32)
    nz = tf.cast(parsed_rec['n_z'], tf.int32)

    # Scaling data at wall

    avg_wall = tfr_path + '/avg_wall.mat'

    print('The inputs are normalized to have a unit Gaussian distribution')

    data = sio.loadmat(avg_wall)

    avgs_wall = tf.constant(data['mean_inputs'].astype(np.float32))
    stds_wall = tf.constant(data['std_inputs'].astype(np.float32))
    inputs1 = filtering(tf.reshape((parsed_rec[f'wall_raw1']-avgs_wall[0])/stds_wall[0],(nz,nx)),filter_size)
    if channels ==2:
        inputs2 = filtering(tf.reshape((parsed_rec[f'wall_raw2']-avgs_wall[1])/stds_wall[1],(nz,nx)), filter_size)
        inputs = tf.stack([inputs1, inputs2], axis = 0)
    if channels == 3:
        inputs2 = filtering(tf.reshape((parsed_rec[f'wall_raw2']-avgs_wall[1])/stds_wall[1],(nz,nx)), filter_size)
        inputs3 = filtering(tf.reshape((parsed_rec[f'wall_raw3']-avgs_wall[2])/stds_wall[2],(nz,nx)), filter_size)
        inputs = tf.stack([inputs1, inputs2,inputs3], axis = 0)

    outputs = parsed_rec['psi'][:n_modes]
    inputs = inputs[:,::filter_size,::filter_size] + order_noise*tf.random.uniform([channels, nz//filter_size, nx//filter_size])
    return inputs, outputs


@tf.function
def tf_parser_training_cpu(rec, tfr_path, channels, n_modes, filter_size):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''
    features = {
        'i_samp': tf.io.FixedLenFeature([], tf.int64),
        'n_x': tf.io.FixedLenFeature([], tf.int64),
        'n_z': tf.io.FixedLenFeature([], tf.int64),
        'wall_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'wall_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'flow_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'psi': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }

    parsed_rec = tf.io.parse_single_example(rec, features)

    nx = tf.cast(parsed_rec['n_x'], tf.int32)
    nz = tf.cast(parsed_rec['n_z'], tf.int32)

    # Scaling data at wall

    avg_wall = tfr_path + '/avg_wall.mat'

    print('The inputs are normalized to have a unit Gaussian distribution')

    data = sio.loadmat(avg_wall)

    avgs_wall = tf.constant(data['mean_inputs'].astype(np.float32))
    stds_wall = tf.constant(data['std_inputs'].astype(np.float32))

    inputs = tf.reshape((parsed_rec['wall_raw1']-avgs_wall[0])/stds_wall[0],(nz, nx, 1))

    for i_comp in range(1, channels):

        inputs = tf.concat((inputs, tf.reshape((parsed_rec[f'wall_raw{i_comp+1}']-avgs_wall[i_comp])/stds_wall[i_comp],(nz, nx, 1))),-1)

    outputs = parsed_rec['psi'][:n_modes]
    inputs = gaussian_filter(inputs,filter_size,nz,nx, channels, kernel)

    return inputs, outputs


def gaussian_filter(A, filter_size, n_z, n_x):
    channel, na, nb = A.shape

    n_z = 64
    n_x = 128
    rows = n_z//filter_size
    cols = n_x//filter_size
    block = tf.zeros([filter_size, filter_size])
    img = np.zeros((channel, rows, cols))
    vector = []
    if filter_size == 2:
        kernel = tf.ones([filter_size, filter_size])
    else:
        F_1 = cv2.getGaussianKernel(ksize=filter_size,sigma=1)
        kernel = F_1*np.transpose(F_1)
    for ch in range(0,channel):
        for j in range(0, n_z - filter_size + 1, filter_size):
            for i in range(0,n_x - filter_size + 1, filter_size):
                block = A[ch, j:j+filter_size,i:i+filter_size]
                block_mean = tf.reduce_sum(block*kernel)
                vector.append(block_mean)

    img = tf.reshape(vector, (channel, rows, cols))
    print(img.shape)
    #img = tf.convert_to_tensor(img)
    return img