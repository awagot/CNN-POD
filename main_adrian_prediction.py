import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 

print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)

if physical_devices:

  try:

    for gpu in physical_devices:

      tf.config.experimental.set_memory_growth(gpu, True)

  except RuntimeError as e:

    print(e)


import time
from models import model_cnn_mlp
from pipelines import generate_default_training_pipeline
from pipelines import generate_downsampling_training_pipeline
from pipelines import generate_meanfilter_training_pipeline
from training import training_loop
import numpy as np

def main():

    """
        Define training pipelines
    """

    if model_name == "downsample2" or model_name == "downsample4" or model_name == "downsample8" or model_name == "downsample16":
        print('Downsampling')
        dataset = generate_downsampling_training_pipeline(tfr_path, channels, n_modes, downsample_value, validation_split, batch_size, shuffle_buffer, n_prefetch, cpu=False)[1]
        n_z = nz//downsample_value
        n_x = nx//downsample_value
    if model_name == "meanfilter2" or model_name == "meanfilter4" or model_name == "meanfilter8" or model_name == "meanfilter16":
        print('Mean Filter')
        dataset = generate_meanfilter_training_pipeline(tfr_path, channels, n_modes, filter_size, validation_split, batch_size, shuffle_buffer, n_prefetch, cpu=False)[1]
        n_z = nz//filter_size
        n_x = nx//filter_size


    model = get_model(channels, n_z, n_x, n_modes, model_name, save_path)
    

    dataset = iter(dataset)
    
    X = np.zeros((batch_size*n_batches,channels,n_z,n_x))
    print(X.shape)
    Y = np.zeros((batch_size*n_batches,n_modes))
    Z = np.zeros((batch_size*n_batches,n_modes))
    for i in range(n_batches): # X are inputs, Y are actual outputs, Z are predicted outputs

        (x, y) = next(dataset)
        z = model.predict(x)
        a = i*batch_size
        print(x.shape, a)

        X[a:a+batch_size,:,:,:] = x
        Y[a:a+batch_size,:] = y
        Z[a:a+batch_size,:] = z
        
    print(Z.shape, X.shape, Y.shape)
    filename = f"{path_mat}{model_name}.npz"
    np.savez(filename, X = X, Y = Y, Z = Z)


    return


def get_model(channels, n_z ,n_x , n_modes, model_name, save_path):
    """
        Define model
    """
    model = model_cnn_mlp(channels, n_z, n_x, n_modes, cpu=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    model_loss = tf.keras.losses.MeanSquaredError()

    """
        Restore Model

    """

    log_folder = f"./logs/"
    checkpoint_dir = f"{save_path}checkpoints_{model_name}"
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    return model




if __name__ == "__main__":

    nz = 64 
    nx = 128 
    n_batches = 1
    epochs = 100
    yp_flow = 15
    n_modes = 10
    channels = 3 
    n_prefetch = 4
    batch_size = 50
    downsample_value = 16
    filter_size = 16
    save_path = ""
    path_mat = "/home/awag/Documents/TFG/MAT/"
    model_name = "downsample16"
    tfr_path = "/home/awag/Documents/TFG/DATA/TFRECORD/D15"
    shuffle_buffer = 5000
    validation_split = 0.2

    main()