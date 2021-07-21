import sys
import json
import os
import pydicom
import datetime

import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import preprocess_input

from dicom_io import read_dicom_uint8, read_dicom_uint16, preprocess_uint8_resnet, preprocess_uint16_inception_input
from img_io import read_jpg_uint8
from image_crop import crop_image_to_mask

def crop_dicom_to_mask(image, mask):
    x = crop_image_to_mask(image, mask,
                               crop_shape=(299, 299),
                               soft_mask=True, crop_masked_img=False)
    x = preprocess_uint16_inception_input(x)
    return x

input_path = sys.argv[1]

IS_DICOM = False

if input_path.endswith('.dcm'):
    IS_DICOM = True

if IS_DICOM:
    object = pydicom.dcmread(fp=input_path, stop_before_pixels=True)
    print('Extracting DICOM metadata...')
    IMG_SHAPE = (object.Rows, object.Columns)

    BITS = object.BitsStored

    try:
        AGE = (datetime.datetime.strptime(object.StudyDate, '%Y%m%d') -
        datetime.datetime.strptime(object.PatientBirthDate, '%Y%m%d')).days/365.24
    except:
        print('WARNING: Age could not be determined from StudyDate tag')
        AGE = -1
else:
    print('WARNING: input is not in DICOM format')
    object = read_jpg_uint8(input_path, target_size=None)
    IMG_SHAPE = object.shape
    AGE = -1
    BITS = 8

if (IMG_SHAPE[0] < 1500) | (IMG_SHAPE[1] < 1500):
    print('WARNING: image resolution is lower than 1500x1500. Interpret with caution!')

if BITS < 16:
    print('WARNING: BitsStored is under 16, falling back to 8-Bit inference. Interpret with caution!')

dicom_metadata = {'age': AGE,
               'rows': IMG_SHAPE[0],
               'columns': IMG_SHAPE[1],
               'bits_stored': BITS}

print('Performing QC...')
body_part = tf.keras.models.load_model('models/inceptionv3_299_299_other-chest', compile=False)
frontal_lateral = tf.keras.models.load_model('models/inceptionv3_299_299_frontal_lateral', compile=False)
view_position = tf.keras.models.load_model('models/inceptionv3_299_299_ap-pa', compile=False)

if IS_DICOM:
    x = read_dicom_uint8(input_path, target_size=(299, 299), preprocess_fn=preprocess_input)
else:
    x = read_jpg_uint8(input_path, target_size=(299, 299), preprocess_fn=preprocess_input)
    x = tf.reshape(x, (1, 299, 299, 3))

proba_body_part = body_part.predict(x)
proba_frontal_lateral = frontal_lateral.predict(x)
proba_view_position = view_position.predict(x)

image_metadata = {'chest_xr_proba': proba_body_part[0].tolist()[1],
                  'frontal_xr_proba': proba_frontal_lateral[0].tolist()[0],
                  'ap_xr_proba': proba_view_position[0].tolist()[0]}

print('Extracting lung segments')

cxr_segmentation = tf.keras.models.load_model('models/resnet50_unet_segmentation_covix')
if IS_DICOM:
    x = read_dicom_uint8(input_path, target_size=(512, 512),
                        preprocess_fn=preprocess_uint8_resnet)
else:
    x = read_jpg_uint8(input_path, target_size=(512, 512),
                        preprocess_fn=preprocess_uint8_resnet)
    x = tf.reshape(x, (1, 512, 512, 3))

proba = cxr_segmentation.predict(x)
proba = proba.reshape((256, 256, 2)).argmax(axis=2)
MASK = np.stack((proba,)*3, axis=-1)
MASK = tf.image.resize(MASK, size=(IMG_SHAPE[0], IMG_SHAPE[1]), method='nearest').numpy() * 255

print('Running CovIx Ensemble...')
mask_model = tf.keras.models.load_model('models/inceptionv3_masks_299_299_diagnosis', compile=False)

if IS_DICOM:
    cxr = read_dicom_uint16(input_path, target_size=(1500, 1500))
else:
    cxr = read_jpg_uint8(input_path, target_size=(1500, 1500))


x = np.array([crop_dicom_to_mask(cxr, MASK) for i in range(50)])
mask_proba = np.mean(mask_model.predict(x), axis=0)

multioutput299 = tf.keras.models.load_model('models/inceptionv3_multi_output_299_299_diagnosis', compile=False)
multioutput764 = tf.keras.models.load_model('models/inceptionv3_multi_output_764_764_diagnosis', compile=False)

if IS_DICOM & (BITS > 8):
    x_299 = read_dicom_uint16(input_path, target_size=(299, 299), preprocess_fn=preprocess_uint16_inception_input)
    x_764 = read_dicom_uint16(input_path, target_size=(764, 764), preprocess_fn=preprocess_uint16_inception_input)
elif IS_DICOM & (BITS < 16):
    x_299 = read_dicom_uint8(input_path, target_size=(299, 299), preprocess_fn=preprocess_input)
    x_764 = read_dicom_uint8(input_path, target_size=(764, 764), preprocess_fn=preprocess_input)

    x_299 = tf.reshape(x_299, (299, 299, 3))
    x_764 = tf.reshape(x_764, (764, 764, 3))
else:
    x_299 = read_jpg_uint8(input_path, target_size=(299, 299), preprocess_fn=preprocess_input)
    x_764 = read_jpg_uint8(input_path, target_size=(764, 764), preprocess_fn=preprocess_input)

proba_multioutput_299 = multioutput299.predict(np.array([x_299]))[1]
proba_multioutput_764 = multioutput764.predict(np.array([x_764]))[1]

ensemble_proba = np.mean(list(np.repeat([mask_proba], 7, axis=0)) +
                list(np.repeat(proba_multioutput_299, 4, axis=0)) +
                list(np.repeat(proba_multioutput_764, 2, axis=0)), axis=0)

proba = {'mask_proba': mask_proba.tolist(),
        'multioutput_299_proba': proba_multioutput_299[0].tolist(),
        'multioutput_764_proba': proba_multioutput_764[0].tolist(),
        'ensemble_proba': ensemble_proba.tolist()}

result = {'input': input_path,
         'dicom_metadata': dicom_metadata,
         'image_metadata': image_metadata,
         'probas': proba,
         'class_names': ['normal', 'abnormal', 'pneumonia', 'covid+']}

print(result)

base_name = os.path.basename(input_path)
with open(base_name+'.json', 'w') as f:
    json.dump(result, f)
