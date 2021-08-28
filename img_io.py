import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

def read_image(path, target_size=(299, 299), preprocess_fn=None):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = image/np.max(image)
    image *= 255
    if target_size is not None:
        image = tf.image.resize(image, size=target_size)
    
    if preprocess_fn is not None:
        image = preprocess_fn(image)
    
    return image