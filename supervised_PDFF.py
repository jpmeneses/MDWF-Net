import tensorflow as tf
import os
import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import (
    mean_squared_error, 
    mean_absolute_error,
    cosine_similarity
)
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    LearningRateScheduler,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.image import image_gradients, total_variation
from tensorflow.keras.experimental import CosineDecay
from skimage.metrics import structural_similarity
from tensorflow.python.client import device_lib

from keras_unet.models import custom_unet
from utils import *
import mebcrn

import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, help="Method for solving the WF-separation problem (MDWF,UNet)")
parser.add_argument('echoes', type=str, help="Map to be displayed (PDFF,R2s,Water)")
args = parser.parse_args()

ech_idx = args.echoes * 2

############################################################
############### DIRECTORIES AND FILENAMES ##################
############################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_onlyMag_2D_mxNorm.hdf5'
dataset_hdf5_2 = 'INTA_GC_192_onlyMag_2D_mxNorm.hdf5'
dataset_hdf5_3 = 'INTArest_GC_192_onlyMag_2D_mxNorm.hdf5'
dataset_hdf5_4 = 'Volunteers_GC_192_onlyMag_2D_mxNorm.hdf5'

model_dir = 'results/'
model_name = 'MDWF_mxNorm_2ech'

############################################################
################### LOAD DATASET 1 #########################
############################################################
f1 = h5py.File(dataset_dir + dataset_hdf5_1, 'r')
acqs_1 = f1['Acquisitions'][...]
out_maps_1 = f1['OutMaps'][...]
f1.close()

idxs_list_1 = []
for nd in range(len(acqs_1)):
  if np.sum(acqs_1[nd,:,:,1])!=0.0:
    idxs_list_1.append(nd)

acqs_1 = acqs_1[idxs_list_1,:,:,:ech_idx]
out_maps_1 = out_maps_1[idxs_list_1,:,:,:]

print('Num. Elements- DS1:', len(acqs_1))

############################################################
################### LOAD DATASET 2 #########################
############################################################
f2 = h5py.File(dataset_dir + dataset_hdf5_2, 'r')
acqs_2 = f2['Acquisitions'][...]
out_maps_2 = f2['OutMaps'][...]
f2.close()

idxs_list_2 = []
for nd in range(len(acqs_2)):
  if np.sum(acqs_2[nd,:,:,1])!=0.0:
    idxs_list_2.append(nd)

acqs_2 = acqs_2[idxs_list_2,:,:,:ech_idx]
out_maps_2 = out_maps_2[idxs_list_2,:,:,:]

print('Num. Elements- DS2:', len(acqs_2))

############################################################
################### LOAD DATASET 3 #########################
############################################################
f3 = h5py.File(dataset_dir + dataset_hdf5_3, 'r')
acqs_3 = f3['Acquisitions'][...]
out_maps_3 = f3['OutMaps'][...]
f3.close()

idxs_list_3 = []
for nd in range(len(acqs_3)):
  if np.sum(acqs_3[nd,:,:,1])!=0.0:
    idxs_list_3.append(nd)

acqs_3 = acqs_3[idxs_list_3,:,:,:ech_idx]
out_maps_3 = out_maps_3[idxs_list_3,:,:,:]

print('Num. Elements- DS3:', len(acqs_3))

############################################################
################### LOAD DATASET 4 #########################
############################################################
f4 = h5py.File(dataset_dir + dataset_hdf5_4, 'r')
acqs_4 = f4['Acquisitions'][...]
out_maps_4 = f4['OutMaps'][...]
f4.close()

idxs_list_4 = []
for nd in range(len(acqs_4)):
  if np.sum(acqs_4[nd,:,:,1])!=0.0:
    idxs_list_4.append(nd)

acqs_4 = acqs_4[idxs_list_4,:,:,:ech_idx]
out_maps_4 = out_maps_4[idxs_list_4,:,:,:]

print('Num. Elements- DS4:', len(acqs_4))

############################################################
################# DATASET PARTITIONS #######################
############################################################

n1_div = 248
n4_div = 434

trainX  = np.concatenate((acqs_1[n1_div:,:,:,:],acqs_3,acqs_4[n4_div:,:,:,:]),axis=0)
valX    = acqs_2
testX   = np.concatenate((acqs_1[:n1_div,:,:,:],acqs_4[:n4_div,:,:,:]),axis=0)

trainY  = np.concatenate((out_maps_1[n1_div:,:,:,:],out_maps_3,out_maps_4[n4_div:,:,:,:]),axis=0)
valY    = out_maps_2
testY   = np.concatenate((out_maps_1[:n1_div,:,:,:],out_maps_4[:n4_div,:,:,:]),axis=0)

# Overall dataset statistics
n_train,hgt,wdt,d_ech = np.shape(trainX)
_,_,_,n_out = np.shape(trainY)
echoes = int(d_ech/2)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',echoes)
print('Output Maps:',n_out)

# Input and output dimensions (training data)
print('Training input shape:',trainX.shape)
print('Training output shape:',trainY.shape)

# Input and output dimensions (validations data)
print('Validation input shape:',valX.shape)
print('Validation output shape:',valY.shape)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

############################################################
############## ADJUST FIELDMAP (U-Net only) ################
############################################################
if args.method == 'UNet':
  trainY[:,:,:,-1]    = 0.5*trainY[:,:,:,-1] + 0.5
  valY[:,:,:,-1]      = 0.5*valY[:,:,:,-1] + 0.5
  testY[:,:,:,-1]     = 0.5*testY[:,:,:,-1] + 0.5

############################################################
################# DATA AUGMENTATION ########################
############################################################
data_gen_args = dict(height_shift_range=25,
                     fill_mode='constant',
                     cval=0,
                     horizontal_flip=True,
                     vertical_flip=True)
datagen_X = ImageDataGenerator(**data_gen_args)
datagen_Y = ImageDataGenerator(**data_gen_args)

############################################################
############### BUILD CNN ARCHITECTURE #####################
############################################################

if method == 'UNet':
  model = custom_unet(
    input_shape=(hgt, wdt, 2*echoes),
    num_classes=n_out,
    activation='relu',
    use_batch_norm=True,
    filters=72,
    num_layers=4,
    output_activation='relu')
elif method == 'MDWF':
  model = custom_model(
    input_shape=(hgt, wdt, 2*echoes),
    activation='relu',
    use_batch_norm=True,
    filters=72,
    num_layers = 4,
    output_activation='relu')

############################################################
################### TRAINING CONFIG ########################
############################################################
logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

my_callbacks = [TensorBoard(log_dir=logdir),
                ModelCheckpoint(filepath = model_dir + model_name + '_best',
                                monitor = 'val_mean_squared_error',
                                verbose = 0,
                                save_best_only = True,
                                save_weights_only = False,
                                mode = 'min')]

verbose, epochs, batch_size = 2, 120, 16
opt = Adam(CosineDecay(5e-4,epochs*(n_train//batch_size)))
model.compile(optimizer=opt,
	metrics=['mean_squared_error','cosine_proximity'],
	loss='mean_absolute_error')
model.optimizer.iterations.assign(0)
train_generator = zip(datagen_X.flow(trainX,batch_size=batch_size,seed=1),
                      datagen_Y.flow(trainY,batch_size=batch_size,seed=1))
model.fit_generator(train_generator,
                    steps_per_epoch = n_train//batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=my_callbacks,
                    shuffle=True,
                    validation_data=(valX, valY))
model.save(model_dir + model_name + '_last')
