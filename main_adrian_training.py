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



def main():
    #dataset_train, dataset_valid = generate_default_training_pipeline(tfr_path, channels, n_modes, validation_split, batch_size, shuffle_buffer, n_prefetch, cpu=False)
    """
        Define training pipelines
    """
    if model_name == "downsample2" or model_name == "downsample4" or model_name == "downsample8" or model_name == "downsample16":
        print('Downsampling')
        dataset_train, dataset_valid = generate_downsampling_training_pipeline(tfr_path, channels, n_modes, downsample_value, validation_split, batch_size, shuffle_buffer, n_prefetch, cpu=False)
        n_z = nz//downsample_value
        n_x = nx//downsample_value
    if model_name == "meanfilter2" or model_name == "meanfilter4" or model_name == "meanfilter8" or model_name == "meanfilter16":
        print('Mean Filter')
        dataset_train, dataset_valid = generate_meanfilter_training_pipeline(tfr_path, channels, n_modes, filter_size, validation_split, batch_size, shuffle_buffer, n_prefetch, cpu=False)
        n_z = nz//filter_size
        n_x = nx//filter_size


    """
        Define model
    """
    model = model_cnn_mlp(channels, n_z, n_x, n_modes, cpu=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    model_loss = tf.keras.losses.MeanSquaredError()

    """
        Training loop
    """

    training_loop(dataset_train, dataset_valid, save_path, model_name, model, optimizer, model_loss, epochs)

    # model_name = f"Ret_flow-reconstruction_yp{yp_flow:03d}"

    # start_time = time.time()

    # train_loss = tf.metrics.Mean()
    # valid_loss = tf.metrics.Mean()

    # for epoch in range(1, epochs + 1):

        # train_loss.reset_states()

        # for (X_target, Y_target) in dataset_train:
  
            # pred_loss, mae = model.train_on_batch(X_target, Y_target)
            # train_loss.update_state(pred_loss)
            
        # for (X_target, Y_target) in dataset_valid:
            # valid_pred_loss, valid_mae = model.test_on_batch(X_target, Y_target)
            # valid_loss.update_state(valid_pred_loss)
        
        # end_time = time.time()

        # if epoch > 10:
        
           #  model.optimizer.lr = 0.001 * tf.math.exp(0.1 * (10 - epoch))

        # print(f'Epoch {epoch:04d}/{epochs:04d}, loss: {train_loss.result().numpy()}, val_loss: {valid_loss.result().numpy()}, elapsed time from start: {end_time - start_time}')

    # predictor_name = models_path + model_name + '_predictor.tf'
    # predictor.save(predictor_name)

    return


if __name__ == "__main__":

    nz = 64
    nx = 128
    epochs = 2
    yp_flow = 15
    n_modes = 10
    channels = 3 
    n_prefetch = 4
    batch_size = 50
    save_path = ""
    model_name = "meanfilter16"
    downsample_value = 16
    filter_size = 16
    tfr_path = "/home/awag/Documents/TFG/DATA/TFRECORD/D15"
    shuffle_buffer = 5000
    validation_split = 0.2

    main()