import numpy as np
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import (
    mean_squared_error, 
    mean_absolute_error,
    cosine_similarity
)
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    SpatialDropout2D,
    UpSampling2D,
    Input,
    concatenate,
    multiply,
    add,
    Activation,
)

def custom_loss(acqs,y_pred):
    batch_size,hgt,wdt,_ = y_pred.shape
    n_ech = 6
    r2_sc = 200.0   # HR:150 / GC:200
    fm_sc = 300.0   # HR:300 / GC:400
    pi = tf.constant(np.pi,dtype=tf.complex64)

    # Echo Times - Dim 1x6
    TE_ini = tf.range(start=1.3e-3,limit=12e-3,delta=2.1e-3,dtype=tf.float32)
    TE = tf.complex(TE_ini,0.0)
    aux = tf.tile(tf.expand_dims(TE,1),[1,wdt])
    aux2 = tf.linalg.diag(aux)
    TE_mat = tf.transpose(aux2,perm=[2,1,0])

    # Fat Peaks Constants - Dim 7x1
    f_peaks = tf.constant([[0.0],[-3.80],[-3.40],[-2.60],[-1.94],[-0.39],[0.60]],dtype=tf.complex64)*1e-6*42.58e6*1.5
    a_peaks = tf.constant([[0.0],[0.087],[0.693],[0.128],[0.004],[0.039],[0.048]],dtype=tf.complex64)

    # Sum along the fat peaks' values
    fact_exp = tf.linalg.matmul(f_peaks,tf.expand_dims(TE,0)) # Dim 7x6
    f_term = tf.math.reduce_sum(a_peaks*tf.math.exp(2j*pi*fact_exp),axis=0) # Dim 1x6
    aux3 = tf.tile(tf.expand_dims(f_term,1),[1,wdt])
    aux4 = tf.linalg.diag(aux3)
    ft_mat = tf.transpose(aux4,perm=[2,1,0])

    # Turn Maps Values to Complex
    y_pred_comp = tf.complex(y_pred,0.0)

    # Signal for predicted value
    # W_pred = y_pred_comp[:,:,:,0]
    W_pred = tf.complex(y_pred[:,:,:,0],y_pred[:,:,:,1])
    W_pred = tf.tile(tf.expand_dims(W_pred,-1),[1,1,1,n_ech])
    # F_pred = y_pred_comp[:,:,:,1]
    F_pred = tf.complex(y_pred[:,:,:,2],y_pred[:,:,:,3])
    F_pred = tf.tile(tf.expand_dims(F_pred,-1),[1,1,1,wdt])
    r2_orig = y_pred_comp[:,:,:,4]
    r2_pred = tf.tile(tf.expand_dims(r2_orig,-1),[1,1,1,wdt])
    fm_orig = y_pred_comp[:,:,:,5]
    fm_pred = tf.tile(tf.expand_dims(fm_orig,-1),[1,1,1,wdt])
    pred_1 = tf.math.exp(-r2_sc*tf.linalg.matmul(r2_pred,TE_mat))     # Dim 1x6
    pred_2 = tf.math.exp(2j*pi*fm_sc*tf.linalg.matmul(fm_pred,TE_mat)) # Dim 1x6
    pred_3 = (W_pred + tf.linalg.matmul(F_pred,ft_mat))                 # Dim 1x6
    In_pred = pred_1 * pred_2 * pred_3 #* tf.math.exp(-1j*pi/2.0)

    # Pre-process acquisitions
    acqs_re = acqs[:,:,:,0::2]
    acqs_im = acqs[:,:,:,1::2]
    acqs_sort = tf.concat([acqs_re,acqs_im],axis=-1)

    # Post-process model reconstructed acquisitions
    Re_pred = tf.math.real(In_pred)
    Im_pred = tf.math.imag(In_pred)
    res_pred = tf.concat([Re_pred,Im_pred],axis=-1)

    # Custom loss
    alpha_r2 = 9e-9
    alpha_fm1 = 2e-8
    # alpha_fm2 = 1e-6
    # regul = alpha_r2*tf.abs(y_pred[:,:,:,4]) + alpha_fm*tf.abs(y_pred[:,:,:,5])
    r2_f32 = tf.expand_dims(y_pred[:,:,:,4],axis=-1)
    fm_f32 = tf.expand_dims(y_pred[:,:,:,5],axis=-1)
    norm_r2 = tf.reduce_mean(tf.reduce_sum(tf.abs(r2_f32),axis=[0,1,2]))
    # tot_var_r2 = tf.reduce_mean(total_variation(r2_f32))
    # tot_var_fm = tf.reduce_mean(total_variation(fm_f32))
    # im_grad = image_gradients(fm_f32)
    # norm2 = tf.reduce_mean(tf.pow(im_grad[0],2.0) + tf.pow(im_grad[1],2.0))
    norm1 = tf.reduce_mean(tf.reduce_sum(tf.abs(fm_f32),axis=[0,1,2]))
    regul = alpha_r2*norm_r2 + alpha_fm1*norm1
    loss = mean_absolute_error(acqs_sort,res_pred) + regul

    return loss

def upsample(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def conv2d_block(
    inputs,
    use_batch_norm=True,
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
        )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
        )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

def custom_model(
    input_shape,
    activation="relu",
    use_batch_norm=True,
    filters=32,
    num_layers=4,
    output_activation="tanh",
):
    # Build U-Net model
    inputs = tf.keras.Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            activation=activation,
            )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        activation=activation,
        )

    cont = 0
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        if cont < 1:
            x2 = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            x3 = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            x4 = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        elif cont >= 1:
            x2 = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x2)
            x3 = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x3)
            x4 = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x4)

        # Water/Fat decoder
        x2 = concatenate([x2, conv])
        x2 = conv2d_block(
            inputs=x2,
            filters=filters,
            use_batch_norm=use_batch_norm,
            activation=activation,
            )

        # R2* decoder
        x3 = concatenate([x3, conv])
        x3 = conv2d_block(
            inputs=x3,
            filters=filters,
            use_batch_norm=use_batch_norm,
            activation=activation,
            )

        # Field decoder
        x4 = concatenate([x4, conv])
        x4 = conv2d_block(
            inputs=x4,
            filters=filters,
            use_batch_norm=use_batch_norm,
            activation=activation,
            )

        # Update counter
        cont += 1

    y2 = Conv2D(2, (1, 1), activation=output_activation)(x2)
    y3 = Conv2D(1, (1, 1), activation='relu')(x3)
    y4 = Conv2D(1, (1, 1), activation='tanh')(x4)
    outputs = concatenate([y2,y3,y4])

    model = Model(inputs=inputs, outputs=outputs)
    return model

#################################################################################
############################# DISPLAY RESULTS ###################################
#################################################################################

def PDFF_at_ROI(X,left_x,sup_y):
  r1,r2 = sup_y,(sup_y+9)
  c1,c2 = left_x,(left_x+9)
  PDFF_crop = X[r1:r2,c1:c2]
  return np.median(PDFF_crop)

def R2_at_ROI(X,left_x,sup_y):
  r1,r2 = sup_y,(sup_y+9)
  c1,c2 = left_x,(left_x+9)
  R2_crop = X[r1:r2,c1:c2]
  return np.mean(R2_crop)

class IndexTracker(object):
  def __init__(self, fig, ax, X, PDFF_bool, lims, npy_file='slices_crops.npy'):
    self.fig = fig
    self.ax = ax
    ax.set_title('use scroll wheel to navigate images')

    self.X = X
    rows, cols, self.slices = X.shape
    self.ind = 0 # self.slices//2

    try:
      with open(npy_file,'rb') as f:
        frms = np.load(f)
        crops_1 = np.load(f)
        crops_2 = np.load(f)
      self.frms = list(frms)
      self.crops_1 = list(crops_1)
      self.crops_2 = list(crops_2)
    except FileNotFoundError:
      print('No previously existent crops.')
      self.frms = []
      self.crops_1 = []
      self.crops_2 = []
    
    self.flag = False
    self.fflag = False

    self.flag2 = False
    self.fflag2 = False

    self.saveFlag = False
    self.eraseFlag = False

    self.PDFF_bool = PDFF_bool
    vmin, vmax = lims
    self.im = ax.imshow(self.X[:, :, self.ind],vmin=vmin,vmax=vmax)
    self.fig.colorbar(self.im,ax=self.ax)
    self.update()

  def onscroll(self, event):
    # print("%s %s" % (event.button, event.step))
    if event.button == 'up':
      self.ind = (self.ind + 1) % self.slices
    else:
      self.ind = (self.ind - 1) % self.slices
    self.ax.patches = []
    self.fflag = False
    self.fflag2 = False
    self.update()

  def button_press(self, event):
    # print("%s %s" % (event.button, event.step))
    if event.button == 1:
      r_ct = np.round(event.xdata)
      c_ct = np.round(event.ydata)
      self.left_x1 = int(r_ct - 4)
      self.sup_y1 = int(c_ct - 4)
      self.rect_gt_1 = patches.Rectangle((self.left_x1,self.sup_y1),9,9,
        linewidth=1.5,edgecolor='r',facecolor='none')
      self.flag = True
      self.update()
    elif event.button == 3:
      r_ct = np.round(event.xdata)
      c_ct = np.round(event.ydata)
      self.left_x2 = int(r_ct - 4)
      self.sup_y2 = int(c_ct - 4)
      self.rect_gt_2 = patches.Rectangle((self.left_x2,self.sup_y2),9,9,
        linewidth=1.5,edgecolor='orange',facecolor='none')
      self.flag2 = True
      self.update()

  def key_press(self,event):
    if (event.key == 'up') or (event.key == 'down'):
      if event.key == 'up':
        self.ind = (self.ind - 1) % self.slices
      elif event.key == 'down':
        self.ind = (self.ind + 1) % self.slices
      self.ax.patches = []
      self.fflag = False
      self.fflag2 = False
      self.update()
    elif event.key == 'v':
      if self.ind in self.frms:
        idx = self.frms.index(self.ind)
        self.frms.pop(idx)
        self.crops_1.pop(idx)
        self.crops_2.pop(idx)
      self.frms.append(self.ind)
      self.crops_1.append([self.left_x1,self.sup_y1])
      self.crops_2.append([self.left_x2,self.sup_y2])
      self.saveFlag = True
      self.update()
    # - - - - 
    elif event.key == 'b':
      if self.ind in self.frms:
        idx = self.frms.index(self.ind)
        self.frms.pop(idx)
        self.crops_1.pop(idx)
        self.crops_2.pop(idx)
        self.ax.patches = []
        self.fflag = False
        self.fflag2 = False
        self.eraseFlag = True
        self.update()
    # - - - - 

  def update(self):
    self.im.set_data(self.X[:, :, self.ind])
    # - - - - 
    if (self.ind in self.frms) and (not(self.flag)) and (not(self.flag2)):
      idx = self.frms.index(self.ind)
      self.flag = True
      self.flag2 = True
      self.left_x1 = self.crops_1[idx][0]
      self.sup_y1 = self.crops_1[idx][1]
      self.left_x2 = self.crops_2[idx][0]
      self.sup_y2 = self.crops_2[idx][1]
      self.rect_gt_1 = patches.Rectangle((self.left_x1,self.sup_y1),9,9,
        linewidth=1.5,edgecolor='r',facecolor='none')
      self.rect_gt_2 = patches.Rectangle((self.left_x2,self.sup_y2),9,9,
        linewidth=1.5,edgecolor='orange',facecolor='none')
    # - - - -
    if self.flag:
      # Calculate PDFF at ROI
      if self.PDFF_bool:
        self.PDFF = PDFF_at_ROI(self.X[:,:,self.ind],self.left_x1,self.sup_y1)
      else:
        self.PDFF = R2_at_ROI(self.X[:,:,self.ind],self.left_x1,self.sup_y1)
      if self.fflag:
        self.ax.patches = self.ax.patches[-1:]
      self.ax.add_patch(self.rect_gt_1)
      self.ax.patches = self.ax.patches[::-1]
      self.flag = False
      self.fflag = True
    if self.flag2:
      # Calculate PDFF at ROI
      if self.PDFF_bool:
        self.PDFF2 = PDFF_at_ROI(self.X[:,:,self.ind],self.left_x2,self.sup_y2)
      else:
        self.PDFF2 = R2_at_ROI(self.X[:,:,self.ind],self.left_x2,self.sup_y2)
      if self.fflag2:
        self.ax.patches = self.ax.patches[:-1]
      self.ax.add_patch(self.rect_gt_2)
      self.flag2 = False
      self.fflag2 = True
    if self.saveFlag:
      self.ax.set_title('Crops successfully saved!')
      self.saveFlag = False
    elif self.eraseFlag:
      self.ax.set_title('Crops of these frame were successfully deleted')
      self.eraseFlag = False
    elif self.fflag and self.fflag2:
      if self.PDFF_bool:
        self.ax.set_title('PDFF_1 = '+str(np.round(self.PDFF*100,2))+'% - '+
          ' PDFF_2 = '+str(np.round(self.PDFF2*100,2))+'%')
      else:
        self.ax.set_title('R2*_1 = '+str(np.round(self.PDFF,2))+'[1/s] - '+
          ' R2*_2 = '+str(np.round(self.PDFF2,2))+'[1/s]')
    else:
      self.ax.set_title('use scroll wheel to navigate images')
    self.ax.set_ylabel('slice %s' % self.ind)
    self.im.axes.figure.canvas.draw()
