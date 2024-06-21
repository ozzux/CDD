from seg_ind import seg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


res = seg('convdata/MS/MS_5.png')/255

model = tf.keras.models.load_model('model.keras')