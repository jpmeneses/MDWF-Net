import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import h5py
import xlsxwriter
from skimage.metrics import structural_similarity

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

# ==============================================================================
# =                                   excel                                    =
# ==============================================================================
workbook = xlsxwriter.Workbook('Radiology models/PDFF-metrics.xlsx')
ws_metrics = workbook.add_worksheet('PDFF metrics')
ws_metrics.write(0,0,'RMSE')
ws_metrics.write(0,1,'MAE')
ws_metrics.write(0,2,'SSIM')

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
G_model = 'UNet' # options: MDWF, UNet, MEBCRN
n_echoes = 3
ech_idx = n_echoes * 2

############################################################
############### DIRECTORIES AND FILENAMES ##################
############################################################
dataset_dir = '../MRI-Datasets/'
dataset_hdf5_1 = 'UNet-JGalgani/JGalgani_GC_192_onlyMag_2D.hdf5'
dataset_hdf5_2 = 'UNet-INTA/INTA_GC_192_onlyMag_2D.hdf5'
dataset_hdf5_3 = 'UNet-INTA_rest/INTArest_GC_192_onlyMag_2D.hdf5'
dataset_hdf5_4 = 'UNet-Volunteers/Volunteers_GC_192_onlyMag_2D.hdf5'

model_load_dir = 'Radiology models/'
model_load = 'UNet_RD_192_3ech_v2_last'

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
# f2 = h5py.File(dataset_dir + dataset_hdf5_2, 'r')
# acqs_2 = f2['Acquisitions'][...]
# out_maps_2 = f2['OutMaps'][...]
# f2.close()

# idxs_list_2 = []
# for nd in range(len(acqs_2)):
#   if np.sum(acqs_2[nd,:,:,1])!=0.0:
#     idxs_list_2.append(nd)

# acqs_2 = acqs_2[idxs_list_2,:,:,:]
# out_maps_2 = out_maps_2[idxs_list_2,:,:,:]

# print('Num. Elements- DS2:', len(acqs_2))

############################################################
################### LOAD DATASET 3 #########################
############################################################
# f3 = h5py.File(dataset_dir + dataset_hdf5_3, 'r')
# acqs_3 = f3['Acquisitions'][...]
# out_maps_3 = f3['OutMaps'][...]
# f3.close()

# idxs_list_3 = []
# for nd in range(len(acqs_3)):
#   if np.sum(acqs_3[nd,:,:,1])!=0.0:
#     idxs_list_3.append(nd)

# acqs_3 = acqs_3[idxs_list_3,:,:,:]
# out_maps_3 = out_maps_3[idxs_list_3,:,:,:]

# print('Num. Elements- DS3:', len(acqs_3))

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

n1_div = 248 # 65
n4_div = 434 # 113

# trainX  = np.concatenate((acqs_1[n1_div:,:,:,:],acqs_3,acqs_4[n4_div:,:,:,:]),axis=0)
# valX    = acqs_2
testX   = np.concatenate((acqs_1[:n1_div,:,:,:],acqs_4[:n4_div,:,:,:]),axis=0)

# trainY  = np.concatenate((out_maps_1[n1_div:,:,:,:],out_maps_3,out_maps_4[n4_div:,:,:,:]),axis=0)
# valY    = out_maps_2
testY   = np.concatenate((out_maps_1[:n1_div,:,:,:],out_maps_4[:n4_div,:,:,:]),axis=0)

# Overall dataset statistics
len_dataset,hgt,wdt,d_ech = np.shape(testX)
_,_,_,n_out = np.shape(testY)
echoes = int(d_ech/2)
r2_sc,fm_sc = 200,300

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',echoes)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,testY))
A_B_dataset_test.batch(1)

# model
with tf.device('/cpu:0'):
    model_ld = keras.models.load_model(model_load_dir + model_load)

@tf.function
def sample_A2B(A):
    indx_FM =tf.concat([tf.zeros_like(A[:,:,:,:3],dtype=tf.int32),
                        tf.ones_like(A[:,:,:,:1],dtype=tf.int32)],axis=-1)
    A2B = model_ld(A, training=False)
    if G_model == 'UNet':
        orig_shape = A2B.shape
        A2B_rest, A2B_FM = tf.dynamic_partition(A2B,indx_FM,num_partitions=2)
        A2B_rest = tf.reshape(A2B_rest,A[:,:,:,:3].shape)
        A2B_FM = tf.reshape(A2B_FM,A[:,:,:,:1].shape)
        A2B_FM = (A2B_FM - 0.5) * 2
        A2B = tf.concat([A2B_rest,A2B_FM],axis=-1)
    A2B = tf.where(A[:,:,:,:4]!=0.0,A2B,0.0)
    return A2B

# run
save_dir = 'Radiology models/samples_testing/PDFF'
mkdir(save_dir)
i = 0

for A, B in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
    A = tf.expand_dims(A,axis=0)
    B = tf.expand_dims(B,axis=0)
    A2B = sample_A2B(A)
    # for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
    #     img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
    #     im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
    #     i += 1

    fig, ((ax1,ax2))=plt.subplots(figsize=(10, 4),nrows=1,ncols=2)
    # Ground truth in the first row
    w_aux = np.squeeze(A2B[:,:,:,0])
    f_aux = np.squeeze(A2B[:,:,:,1])
    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0
    PDFF_ok = ax1.imshow(PDFF_aux, cmap='bone',
                      interpolation='none', vmin=0, vmax=1)
    fig.colorbar(PDFF_ok, ax=ax1)
    ax1.axis('off')

    # Computed maps in the second row
    wn_aux = np.squeeze(B[:,:,:,0])
    fn_aux = np.squeeze(B[:,:,:,1])
    PDFFn_aux = fn_aux/(wn_aux+fn_aux)
    PDFFn_aux[np.isnan(PDFFn_aux)] = 0.0
    # PDFF_model = ax2.imshow(PDFFn_aux, cmap='bone',
    #                     interpolation='none', vmin=0, vmax=1)
    # fig.colorbar(PDFF_model, ax=ax2)
    # ax2.axis('off')

    # # plt.show()
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #             hspace = 0.1, wspace = 0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig(save_dir+'/sample'+str(i).zfill(3)+'.png',bbox_inches = 'tight',
    #     pad_inches = 0)
    # plt.close(fig)

    # Export to Excel file
    # MSE
    RMSE_PDFF = np.sqrt(np.mean(tf.square(PDFF_aux-PDFFn_aux), axis=(0,1)))
    MAE_PDFF = np.mean(tf.abs(PDFF_aux-PDFFn_aux), axis=(0,1))
    PDFF_ssim = structural_similarity(PDFF_aux,PDFFn_aux,multichannel=False)

    ws_metrics.write(i+1,0,RMSE_PDFF)
    ws_metrics.write(i+1,1,MAE_PDFF)
    ws_metrics.write(i+1,2,PDFF_ssim)
    
    i += 1

workbook.close()