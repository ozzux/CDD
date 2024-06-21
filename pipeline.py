from seg_ind import seg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

res = seg('convdata/MS/MS_5.png')/255

model = keras.models.load_model('model.h5')
model.predict(res)
