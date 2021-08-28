import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

def read_dicom_uint8(path, target_size=(299, 299), preprocess_fn=None):
    image = tf.io.read_file(path)
    image = tfio.image.decode_dicom_image(image, color_dim=True,
                                          dtype=tf.uint8, scale='auto',
                                          on_error='lossy')
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.uint8)
    image = tf.image.resize(image, size=target_size)
    
    if preprocess_fn is not None:
        image = preprocess_fn(image)
    
    return image

def read_dicom_uint16(path, target_size=(299, 299), preprocess_fn=None):
    image = tf.io.read_file(path)
    image = tfio.image.decode_dicom_image(image, color_dim=True,
                                          dtype=tf.uint16, scale='auto',
                                          on_error='strict')[0]
    image = tf.image.grayscale_to_rgb(tf.cast(image, tf.int32))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, size=target_size)
    
    if preprocess_fn is not None:
        image = preprocess_fn(image)
    
    return image

def preprocess_uint8_resnet(x):
    x = tf.cast(x, tf.float32)
    mean = [123.68, 116.779, 103.939]
    mean_tensor = tf.keras.backend.constant(-np.array(mean))
    x = tf.keras.backend.bias_add(x, mean_tensor, tf.keras.backend.image_data_format())
    return x

def preprocess_uint16_inception_input(x):
    x /= 32767.5
    x -= 1
    return x

def preprocess_arbitrary_inception_input(x):
    res = (2 * (x - np.min(x))/(np.max(x) - np.min(x))) - 1
    return res