import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

class Model_vault:
  def __init__(self, sequence_len=157, mode='hr'):
    self.sequence_len = sequence_len
    self.model_names=[]
    self.mode = mode

  def _standard_scaler(self,input_tensor):
    # (input - mean) / std
    if self.mode == 'hr':
      y = (input_tensor - tf.expand_dims(tf.repeat(tf.math.reduce_mean(input_tensor, 1),self.sequence_len,1),2)) 
      y = y / tf.expand_dims(tf.repeat(tf.math.reduce_std(input_tensor, 1),self.sequence_len,1),2)
      return y
    else:
      y = (input_tensor - tf.repeat(tf.expand_dims(tf.math.reduce_mean(input_tensor, 1),1),self.sequence_len,1))
      y = y / tf.repeat(tf.expand_dims(tf.math.reduce_std(input_tensor, 1),1),self.sequence_len,1)
      return y

  def squeeze_excite_block(self, input, ratio=16, filters=32):
    init = input
    channel_axis = -1
    se_shape = (1, filters)

    se = layers.GlobalAveragePooling1D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = layers.Permute((3, 1, 2))(se)

    x = layers.multiply([init, se])
    return x

  def create_base_model(self):
    x = None
    if self.mode =='hr':
      x = layers.Input(shape=(self.sequence_len,1))
    elif self.mode =='spo2':
      x = layers.Input(shape=(self.sequence_len,3))

    y = self._standard_scaler(x)
    y = layers.Conv1D(8, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.BatchNormalization(axis=1)(y)
    y = layers.MaxPooling1D()(y)
    y = layers.Conv1D(8, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.BatchNormalization(axis=1)(y)
    y = layers.MaxPooling1D()(y)
    y = layers.Conv1D(16, kernel_size=1, strides=1, activation='relu') (y)
    y = layers.BatchNormalization(axis=1)(y)
    y = layers.Conv1D(24, kernel_size=1, strides=1, activation='relu') (y)
    y = layers.BatchNormalization(axis=1)(y)
    y = layers.Dropout(0.25)(y)
    y = layers.Flatten()(y)
    y = layers.Dense(32,activation='linear')(y)
    y = layers.Dense(1)(y)
    self.model_names.append("BASE")

    return Model(inputs=x, outputs=y)

  def create_fcn(self):
    x = None
    if self.mode =='hr':
      x = layers.Input(shape=(self.sequence_len,1))
    elif self.mode =='spo2':
      x = layers.Input(shape=(self.sequence_len,3))
    
    y = self._standard_scaler(x)
    y = layers.Conv1D(8, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.MaxPooling1D()(y)
    y = layers.Conv1D(8, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.MaxPooling1D()(y)
    y = layers.Conv1D(16, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.Conv1D(24, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.Conv1D(32, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.Conv1D(48, kernel_size=3, strides=1, activation='relu') (y)
    y = layers.Conv1D(1, kernel_size=3, strides=1) (y)
    y = layers.GlobalAveragePooling1D()(y)
    self.model_names.append("FCN")

    return Model(inputs=x, outputs=y)

  def _residual_block(self, input_tensor, num_filt, num_inp_filt):
    y = layers.Conv1D(num_filt, kernel_size=3, strides=1, activation='elu',padding='same') (input_tensor)
    y = layers.Conv1D(num_inp_filt, kernel_size=3, strides=1, activation='elu',padding='same') (y)
    return input_tensor + y

  def _e_residual_block(self, input_tensor, num_filt, num_inp_filt):
    y = layers.Conv1D(num_inp_filt, kernel_size=3, strides=1, activation='elu',padding='same') (input_tensor)
    y = layers.Conv1D(num_filt, kernel_size=1, strides=1, activation='elu',padding='same') (y)
    y = layers.Conv1D(num_inp_filt, kernel_size=1, strides=1, activation='elu',padding='same') (y)
    return input_tensor + y

  def create_fcn_residual(self):
    x = None
    if self.mode =='hr':
      x = layers.Input(shape=(self.sequence_len,1))
    elif self.mode =='spo2':
      x = layers.Input(shape=(self.sequence_len,3))
    
    y = self._standard_scaler(x)
    y = layers.Conv1D(8, kernel_size=3, strides=1, activation='elu') (y)
    y = layers.MaxPooling1D()(y)
    y = self._residual_block(y, 16, 8)
    y = layers.Conv1D(16, kernel_size=3, strides=1, activation='elu') (y)
    y = self._residual_block(y, 24, 16)
    y = layers.Conv1D(24, kernel_size=3, strides=1, activation='elu') (y)
    y = self._residual_block(y, 32, 24)
    y = layers.Conv1D(32, kernel_size=3, strides=1, activation='elu') (y)
    y = self._residual_block(y, 48, 32)
    y = layers.Conv1D(48, kernel_size=3, strides=1, activation='elu') (y)
    y = self._residual_block(y, 64, 48)
    y = layers.Conv1D(64, kernel_size=3, strides=1, activation='elu') (y)
    y = layers.Conv1D(1, kernel_size=3, strides=1) (y)
    y = layers.GlobalAveragePooling1D()(y)
    self.model_names.append("FCN_Residual")

    return Model(inputs=x, outputs=y)

  def create_fcn_dct(self, single=False):
    x = None
    if self.mode =='hr':
      x = layers.Input(shape=(self.sequence_len,1))
    elif self.mode =='spo2':
      x = layers.Input(shape=(self.sequence_len,3))
  
    y = self._standard_scaler(x)
    f = tf.signal.dct(y)
    f = tf.gather(f,indices=[i for i in range(3,100)],axis=1)
    y = layers.SeparableConv1D(4, kernel_size=3, strides=1, padding='same', activation='relu') (f)
    y = layers.SeparableConv1D(6, kernel_size=3, strides=1, padding='same', activation='relu') (y)
    y = layers.Conv1D(8, kernel_size=3, strides=1,  padding='same',activation='elu') (y)
    y = layers.Conv1D(8, kernel_size=3, strides=1, activation='elu') (y)
    y = layers.Conv1D(12, kernel_size=7, strides=1, activation='elu') (y)
    y = layers.Conv1D(16, kernel_size=5, strides=1, activation='elu') (y)
    y = layers.Conv1D(24, kernel_size=3, strides=1, activation='elu') (y)
    y = layers.Conv1D(1, kernel_size=1, strides=1) (y)
    y = layers.GlobalAveragePooling1D()(y)
    self.model_names.append("FCN_DCT")

    if not single:
      return Model(inputs=x, outputs=y)
    else:
      return [Model(inputs=x, outputs=y)], self.model_names

  def create_all_models(self):
    models = [self.create_base_model(), self.create_fcn(), self.create_fcn_residual(), self.create_fcn_dct()]
    return models, self.model_names