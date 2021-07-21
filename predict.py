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
from image_crop import crop_image_to_mask

def crop_dicom_to_mask(image, mask):
    x = crop_image_to_mask(image, mask,
                               crop_shape=(299, 299),
                               soft_mask=True, crop_masked_img=False)
    x = preprocess_uint16_inception_input(x)

    return x

dicom_path = sys.argv[1]
dcm = pydicom.dcmread(fp=dicom_path, stop_before_pixels=True)

print('Extracting DICOM metadata...')
IMG_SHAPE = (dcm.Rows, dcm.Columns)

try:
    age = (datetime.datetime.strptime(dcm.StudyDate, '%Y%m%d') -
    datetime.datetime.strptime(dcm.PatientBirthDate, '%Y%m%d')).days/365.24
except:
    print('WARNING: Age could not be determined from StudyDate tag')
    age = -1

dicom_metadata = {'age': age,
               'rows': dcm.Rows,
               'columns': dcm.Columns,
               'bits_stored': dcm.BitsStored}

print('Performing QC...')
body_part = tf.keras.models.load_model('models/inceptionv3_299_299_other-chest', compile=False)
frontal_lateral = tf.keras.models.load_model('models/inceptionv3_299_299_frontal_lateral', compile=False)
view_position = tf.keras.models.load_model('models/inceptionv3_299_299_ap-pa', compile=False)

x = read_dicom_uint8(dicom_path, target_size=(299, 299), preprocess_fn=preprocess_input)
proba_body_part = body_part.predict(x)
proba_frontal_lateral = frontal_lateral.predict(x)
proba_view_position = view_position.predict(x)

image_metadata = {'chest_xr_proba': proba_body_part[0].tolist()[1],
                  'frontal_xr_proba': proba_frontal_lateral[0].tolist()[0],
                  'ap_xr_proba': proba_view_position[0].tolist()[0]}

print('Extracting lung segments')

cxr_segmentation = tf.keras.models.load_model('models/resnet50_unet_segmentation_covix')
x = read_dicom_uint8(dicom_path, target_size=(512, 512), preprocess_fn=preprocess_uint8_resnet)
proba = cxr_segmentation.predict(x)
proba = proba.reshape((256, 256, 2)).argmax(axis=2)
MASK = np.stack((proba,)*3, axis=-1)
MASK = tf.image.resize(MASK, size=IMG_SHAPE, method='nearest').numpy() * 255

print('Running CovIx Ensemble...')
mask_model = tf.keras.models.load_model('models/inceptionv3_masks_299_299_diagnosis', compile=False)
cxr = read_dicom_uint16(dicom_path, target_size=(1500, 1500))
x = np.array([crop_dicom_to_mask(cxr, MASK) for i in range(50)])
mask_proba = np.mean(mask_model.predict(x), axis=0)

multioutput299 = tf.keras.models.load_model('models/inceptionv3_multi_output_299_299_diagnosis', compile=False)
multioutput764 = tf.keras.models.load_model('models/inceptionv3_multi_output_764_764_diagnosis', compile=False)
x_299 = read_dicom_uint16(dicom_path, target_size=(299, 299), preprocess_fn=preprocess_uint16_inception_input)
x_764 = read_dicom_uint16(dicom_path, target_size=(764, 764), preprocess_fn=preprocess_uint16_inception_input)

proba_multioutput_299 = multioutput299.predict(np.array([x_299]))[1]
proba_multioutput_764 = multioutput299.predict(np.array([x_764]))[1]

ensemble_proba = np.mean(list(np.repeat([mask_proba], 7, axis=0)) +
                list(np.repeat(proba_multioutput_299, 4, axis=0)) +
                list(np.repeat(proba_multioutput_764, 2, axis=0)), axis=0)

proba = {'mask_proba': mask_proba.tolist(),
        'multioutput_299_proba': proba_multioutput_299[0].tolist(),
        'multioutput_764_proba': proba_multioutput_764[0].tolist(),
        'ensemble_proba': ensemble_proba.tolist()}

result = {'input': dicom_path,
         'dicom_metadata': dicom_metadata,
         'image_metadata': image_metadata,
         'probas': proba,
         'class_names': ['normal', 'abnormal', 'pneumonia', 'covid+']}

print(result)

base_name = os.path.basename(dicom_path)
with open(base_name+'.json', 'w') as f:
    json.dump(result, f)
