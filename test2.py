import os
import tensorflow as tf
from models import model_fcn

channels = 3
n_z = 160
n_x = 320
n_modes = 100

model = model_fcn(channels, n_z, n_x, n_modes, cpu=False)