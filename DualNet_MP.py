# -*- coding: utf-8 -*-
'''
Created on Sun Nov  1 16:24:41 2020

@author: user
'''

from __future__ import print_function
import math
from ccnn_layers import CConv2D, CConv2D_getlayer
from tensorflow.keras.layers import InputSpec, Layer, Lambda, LeakyReLU, PReLU, ReLU, Dense, Conv2D, BatchNormalization, Activation, Add, Subtract, Dropout
from tensorflow.keras.layers import Multiply, Input, Flatten, Reshape, Dot, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau

from scipy.io import loadmat
import numpy as np
import os
import h5py
import scipy.io as io
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
if __name__ == '__main__':
    import argparse
    import sys
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-nem', '--n_epoch_mag', type=int, default=1000, help='number of epochs to train magnitude branch')
    parser.add_argument('-nep', '--n_epoch_pha', type=int, default=1000, help='number of epochs to train the part other than magnitude branch')
    parser.add_argument('-fs', '--from_scratch', type=int, default=0, help='re-train or not')
    parser.add_argument('-fspo', '--from_scratch_phase_only', type=int, default=0, help='re-train or not')
    parser.add_argument('-b', '--n_batch', type=int, default=200, help='number of batches to fit on (ignored during debug mode)')
    parser.add_argument('-crm', '--CR_MAG', type=int, default=4, help='compression factor for MAG')
    parser.add_argument('-crp', '--CR_PHA', type=int, default=8, help='compression factor for PHA')
    parser.add_argument('-qb', '--QB', type=int, default=8, help='number of qunatization bits')
    parser.add_argument('-env', '--scen', type=str, default='indoor', help='scenario, (i.e., outdoor or indoor)')
    parser.add_argument('-loss', '--loss', type=str, default='SMAPE', help='type of loss function, not available to use other input so far')
    parser.add_argument('-ld', '--ld', type=str, default='CCNN', help='core layer design, not available to use other input so far')
    parser.add_argument('-QT', '--QT', type=str, default='SSQ', help='quantization type, not available to use other input so far')
    opt = parser.parse_args()
# model parameter settings
    width = 32
    height = 32
    number_channel = 1
    size_codeword_MAG = width*height/opt.CR_MAG
    size_codeword_PHA = width*height/opt.CR_PHA
    
    batch_size = opt.n_batch # orig paper trained all networks with batch_size=128
    # data loading starts
    min_max_scaler = preprocessing.MinMaxScaler()
    print('data is loading ... ')
    if opt.scen == 'outdoor':
        mat2 = loadmat('H_outdoor.mat')
        x_train_DL = mat2['H_dl']
        x_train_UL = mat2['H_ul']
        y_train_DL = mat2['H_dl']
    else:
        mat2 = loadmat('H_indoor.mat')
        x_train_DL = mat2['Hur_down_t1']
        x_train_UL = mat2['Hur_up_t1']
        y_train_DL = mat2['Hur_down_t1']
        x_train_DL = np.concatenate((x_train_DL[:,16:32],x_train_DL[:,0:16]),axis = 1)
        x_train_UL = np.concatenate((x_train_UL[:,16:32],x_train_UL[:,0:16]),axis = 1)
        y_train_DL = np.concatenate((y_train_DL[:,16:32],y_train_DL[:,0:16]),axis = 1)
    print('data loading is finised\n')
    print('data processing')
    x_test_DL = x_train_DL[50000:70000,]
    x_test_UL = x_train_UL[50000:70000,]
    y_test = y_train_DL[50000:70000,]
    
    x_train_DL = np.concatenate((x_train_DL[0:50000,],x_train_DL[70000:100000,]),axis=0)
    x_train_UL = np.concatenate((x_train_UL[0:50000,],x_train_UL[70000:100000,]),axis=0)
    y_train_DL = np.concatenate((y_train_DL[0:50000,],y_train_DL[70000:100000,]),axis=0)
    
    x_train_DL = np.expand_dims(x_train_DL,-1)
    x_train_UL = np.expand_dims(x_train_UL,-1)
    y_train_DL = np.expand_dims(y_train_DL,-1)
    tmp_complex = np.concatenate((x_train_DL,x_train_UL,y_train_DL),axis = 0)
    x_train_DL = np.absolute(x_train_DL)
    x_train_UL = np.absolute(x_train_UL)
    y_train_DL = np.absolute(y_train_DL)
    tmp = np.concatenate((x_train_DL,x_train_UL,y_train_DL),axis = 0)
    tmp = np.reshape(tmp, (np.size(tmp,axis=0),width*height))
    tmp = np.transpose(tmp)
    tmp = min_max_scaler.fit_transform(tmp)
    tmp = np.transpose(tmp)
    tmp = np.reshape(tmp, (np.size(tmp,axis=0),width,height,1))
    tmp_phase = np.angle(tmp_complex)
    tmp_phase_cos = np.cos(tmp_phase)
    tmp_real = np.multiply(tmp,np.cos(tmp_phase))
    tmp_imag = np.multiply(tmp,np.sin(tmp_phase))
    x_train_DL = tmp[0:np.size(x_train_DL,axis=0),:,:,0]
    x_train_DL_phase_cos = tmp_phase_cos[0:np.size(x_train_DL,axis=0),:,:,0]
    x_train_DL_phase = tmp_phase[0:np.size(x_train_DL,axis=0),:,:,0]
    x_train_UL = tmp[np.size(x_train_UL,axis=0):2*np.size(x_train_UL,axis=0),:,:,0]
    y_train_DL = tmp[2*np.size(y_train_DL,axis=0):3*np.size(y_train_DL,axis=0),:,:,0]
    y_train_DL_phase_cos = tmp_phase_cos[2*np.size(y_train_DL,axis=0):3*np.size(y_train_DL,axis=0),:,:,0]
    y_train_DL_phase = tmp_phase[2*np.size(y_train_DL,axis=0):3*np.size(y_train_DL,axis=0),:,:,0]
    y_train_DL_real = tmp_real[2*np.size(y_train_DL,axis=0):3*np.size(y_train_DL,axis=0),:,:,0]
    y_train_DL_imag = tmp_imag[2*np.size(y_train_DL,axis=0):3*np.size(y_train_DL,axis=0),:,:,0]
    x_train_DL = np.expand_dims(x_train_DL,-1)
    x_train_DL_phase_cos = np.expand_dims(x_train_DL_phase_cos,-1)
    x_train_DL_phase = np.expand_dims(x_train_DL_phase,-1)
    x_train_UL = np.expand_dims(x_train_UL,-1)
    y_train_DL = np.expand_dims(y_train_DL,-1)
    y_train_DL_real = np.expand_dims(y_train_DL_real,-1)
    y_train_DL_imag = np.expand_dims(y_train_DL_imag,-1)
    y_train_DL_phase_cos = np.expand_dims(y_train_DL_phase_cos,-1)
    y_train_DL_phase = np.expand_dims(y_train_DL_phase,-1)
    sipping_matrix = x_train_DL>=0.0
    sipping_matrix = sipping_matrix.astype(float)
    x_train_phase_cos = np.multiply(x_train_DL_phase_cos,sipping_matrix)
    x_train_DL_phase_bit = x_train_DL_phase >= 0
    x_train_DL_phase_bit = x_train_DL_phase_bit.astype(float)*2 - 1
    x_train_DL_phase = np.concatenate((x_train_phase_cos,x_train_DL_phase_bit),axis = -1)
    y_train_DL_final = np.concatenate((y_train_DL_real,y_train_DL_imag),axis = -1)
    
    def DualNetMAG_layers(input1,input2):
        x = CConv2D(16, kernel_size=(7,7), padding='valid')(input1)
        x = LeakyReLU()(x)
        x = CConv2D(8, kernel_size=(7,7), padding='valid')(x)
        x = LeakyReLU()(x)
        x = CConv2D(4, kernel_size=(7,7), padding='valid')(x)
        x = LeakyReLU()(x)
        x = CConv2D(2, kernel_size=(7,7), padding='valid')(x)
        x = LeakyReLU()(x)
        x = CConv2D(1, kernel_size=(7,7), padding='valid')(x)
        x = ReLU()(x)
        x = Flatten()(x)
        Encoder_outputs = Dense(size_codeword_MAG)(x)
        Decoder_inputs_1 = Encoder_outputs
        x = Dense(width*height*number_channel)(Decoder_inputs_1)
        x_UL = Flatten()(input2)
        x = Concatenate(axis = -1)([x, x_UL])
        x = Reshape(target_shape = (width,height,2))(x)
        x = CConv2D(16, kernel_size=(7,7), padding='valid')(x)
        x = LeakyReLU()(x)
        x = CConv2D(8, kernel_size=(7,7), padding='valid')(x)
        x = LeakyReLU()(x)
        x = CConv2D(4, kernel_size=(7,7), padding='valid')(x)
        x = LeakyReLU()(x)
        x = CConv2D(2, kernel_size=(7,7), padding="valid")(x)
        x = LeakyReLU()(x)
        x = CConv2D(1, kernel_size=(7,7), padding='valid')(x)
        y = Activation('relu')(x)
        return y
    def crop(dimension, start, end):
        # Crops (or slices) a Tensor on a given dimension from start to end
        # example : to crop tensor x[:, :, 5:10]
        # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]
        return Lambda(func)
    def OneGenerate(x):
        return tf.constant(1, dtype = tf.float32, shape = [1, width,height,number_channel])
    def sqrt_(x):
        return K.sqrt(x)
    steepness = 25
    def Qant_function(x):
        sequence = range(-2**(opt.QB-1)+1,2**(opt.QB-1))
        g = tf.math.sigmoid(steepness*(x-(-2**(opt.QB-1))-0.5))
        for iii in sequence:
            g = g + tf.math.sigmoid(steepness*(x-iii-0.5))
        g = g - 2**(opt.QB-1)
        return g
    
    class MultAll(Layer):
        def __init__(self, **kwargs):
            self.axis = -1
            super(MultAll, self).__init__(**kwargs)
    
        def build(self, input_shape):
            if input_shape[-1] is None:
                raise ValueError('Axis ' +  + ' of '
                                 'input tensor should have a defined dimension '
                                 'but the layer received an input with shape ' +
                                 str(input_shape) + '.')
            self.input_spec = InputSpec(ndim=len(input_shape), axes=dict(list(enumerate(input_shape[1:], start=1))))
            self.kernel = self.add_weight(name='kernel', 
                                          shape=input_shape[1:],
                                          initializer='uniform',
                                          trainable=True)
            super(MultAll, self).build(input_shape)  # Be sure to call this at the end
    
        def call(self, x):
            return [tf.multiply(x,self.kernel), self.kernel]
    
        def compute_output_shape(self, input_shape):
            return (input_shape)
    def division(x):
        [x1,x2] = x
        return tf.math.divide(x1,x2)
    def shaper(x):
        [x1, x2] = x
        x1 = tf.reshape(x1, shape = [1,x1.shape[0]])
        return x1
    def ScaleMatrixGenerator(x):
        return tf.constant(2**(opt.QB-1), dtype = tf.float32, shape = [1, int(np.round(size_codeword_PHA*2))])
    
    ################################### MAG model (considering perfect quantization)
    Encoder_inputs = Input(shape = [width, height, number_channel], name ='DL_MAG')
    Decoder_inputs_2 = Input(shape = [width, height, number_channel], name ='UL_MAG')
    Decoder_outputs = DualNetMAG_layers(Encoder_inputs,Decoder_inputs_2)
    DualNetMAG = Model(inputs=[Encoder_inputs, Decoder_inputs_2], outputs = [Decoder_outputs], name = 'DualNetMAG')
    ################################### Phase model
    ### construct phase encoder
    Input_DualNetPhase_Encoder = Input(shape = [width, height, 2], name ='DL_PHASE_EN_Input')
    Encoder_Phase_inputs_clip = crop(3,0,1)(Input_DualNetPhase_Encoder)
    Encoder_Phase_inputs_code = crop(3,1,2)(Input_DualNetPhase_Encoder)
    y = CConv2D(16, kernel_size=(7,7), padding='valid')(Encoder_Phase_inputs_clip)
    y = Activation('tanh')(y)
    y = CConv2D(8, kernel_size=(7,7), padding='valid')(y)
    y = Activation('tanh')(y)
    y = CConv2D(4, kernel_size=(7,7), padding='valid')(y)
    y = Activation('tanh')(y)
    y = CConv2D(2, kernel_size=(7,7), padding='valid')(y)
    y = Activation('tanh')(y)
    y = CConv2D(1, kernel_size=(7,7), padding='valid')(y)
    y = Activation('tanh')(y)
    y = Flatten()(y)
    y = Dense(int(np.round(size_codeword_PHA*2)))(y)
    y = Activation('tanh')(y)
    [Output_DualNetPhase_Encoder, Q_weights] = MultAll(name ='Qinterval')(y)
    Q_weights = Lambda(shaper)([Q_weights,Output_DualNetPhase_Encoder])
    #Q_weights = Qinterval.get_weights()
    DualNetPhase_Encoder = Model(inputs=[Input_DualNetPhase_Encoder], outputs=[Output_DualNetPhase_Encoder, Encoder_Phase_inputs_code, Q_weights], name = 'DualNetPhase_Encoder')
    ####################################################################
    # Quantizer
    ScaleMatrix = Lambda(ScaleMatrixGenerator)([Output_DualNetPhase_Encoder, Output_DualNetPhase_Encoder])
    Output_DualNetPhase_Encoder = Multiply()([ScaleMatrix, Output_DualNetPhase_Encoder])
    Output_Quantizer = Lambda(Qant_function)(Output_DualNetPhase_Encoder)
    Output_Quantizer = Lambda(division)([Output_Quantizer, ScaleMatrix])
    #########################################################################
    # construct phase decoder
    Q_codeword = Output_Quantizer
    codeword = Lambda(division, name = 'DE_0')([Q_codeword, Q_weights])
    y = Dense(width*height, name = 'DE_1')(codeword)
    y = Reshape(target_shape = (width,height,1), name = 'DE_2')(y)
    y = CConv2D(16, kernel_size=(5,5), padding='valid', name = 'DE_5')(y)
    y = Activation('linear', name = 'DE_6')(y)
    y = CConv2D(8, kernel_size=(5,5), padding='valid', name = 'DE_7')(y)
    y = Activation('linear', name = 'DE_8')(y)
    y = CConv2D(4, kernel_size=(5,5), padding='valid', name = 'DE_9')(y)
    y = Activation('linear', name = 'DE_10')(y)
    y = CConv2D(2, kernel_size=(5,5), padding='valid', name = 'DE_9_1')(y)
    y = Activation('tanh', name = 'DE_10_1')(y)
    y = CConv2D(1, kernel_size=(5,5), padding='valid', name = 'DE_11')(y)
    Decoder_Phase_outputs_cos = Activation('tanh', name = 'DE_12')(y)
    One_matrix = Lambda(OneGenerate, name = 'DE_13')(Decoder_Phase_outputs_cos)
    Decoder_Phase_outputs_cos_2 = Multiply(name = 'DE_14')([Decoder_Phase_outputs_cos,Decoder_Phase_outputs_cos])
    Decoder_Phase_outputs_sin_MAG = Subtract(name = 'DE_15')([One_matrix, Decoder_Phase_outputs_cos_2])
    Decoder_Phase_outputs_sin_MAG = Lambda(sqrt_, name = 'DE_16')(Decoder_Phase_outputs_sin_MAG)
    Decoder_Phase_outputs_sin = Multiply(name = 'DE_17')([Decoder_Phase_outputs_sin_MAG, Encoder_Phase_inputs_code])
    Decoder_ini_estimate_real = Multiply(name = 'DE_18')([Decoder_Phase_outputs_cos, Decoder_outputs])
    Decoder_ini_estimate_imag = Multiply(name = 'DE_19')([Decoder_Phase_outputs_sin, Decoder_outputs])
    Decoder_ini_estimate = Concatenate(axis = -1, name = 'DE_20')([Decoder_ini_estimate_real,Decoder_ini_estimate_imag])
    copy = Decoder_ini_estimate
    z = CConv2D(16, kernel_size=(5,5), padding='valid', name = 'DE_21')(Decoder_ini_estimate)
    z = LeakyReLU(name = 'DE_22')(z)
    z = CConv2D(8, kernel_size=(5,5), padding='valid', name = 'DE_23')(z)
    z = LeakyReLU(name = 'DE_24')(z)
    z = CConv2D(4, kernel_size=(5,5), padding='valid', name = 'DE_25')(z)
    z = LeakyReLU(name = 'DE_26')(z)
    z = CConv2D(2, kernel_size=(5,5), padding='valid', name = 'DE_27')(z)
    z = Activation('tanh',name = 'DE_28')(z)
    Output_DualNetPhase_Decoder = Add(name = 'DE_29')([z,copy])
    DualNetFull = Model(inputs=[Encoder_inputs, Decoder_inputs_2, Input_DualNetPhase_Encoder], outputs = [Output_DualNetPhase_Decoder])
    ################################################################################
    ### copy the phase decoder
    Input1_DualNetPhase_Decoder = Input(shape = [int(np.round(size_codeword_PHA*2))], name = 'DL_PHASE_DE_Input1')
    Input2_DualNetPhase_Decoder = Input(shape = [width, height, 1], name = 'DL_PHASE_DE_Input_SIGN')
    Input3_DualNetPhase_Decoder = Input(shape = [width, height, 1], name = 'DL_PHASE_DE_Input_DL_MAG')
    Input4_DualNetPhase_Decoder = Input(shape = [int(np.round(size_codeword_PHA*2))], name = 'DL_PHASE_DE_Q_weight')
    xx = DualNetFull.get_layer('DE_0')([Input1_DualNetPhase_Decoder, Input4_DualNetPhase_Decoder])
    xx = DualNetFull.get_layer('DE_1')(xx)
    xx = DualNetFull.get_layer('DE_2')(xx)
    xx = CConv2D_getlayer(DualNetFull, 16, kernel_size=(5,5), padding='valid', name = 'DE_5')(xx)
    xx = DualNetFull.get_layer('DE_6')(xx)
    xx = CConv2D_getlayer(DualNetFull, 8, kernel_size=(5,5), padding='valid', name = 'DE_7')(xx)
    xx = DualNetFull.get_layer('DE_8')(xx)
    xx = CConv2D_getlayer(DualNetFull, 4, kernel_size=(5,5), padding='valid', name = 'DE_9')(xx)
    xx = DualNetFull.get_layer('DE_10')(xx)
    xx = CConv2D_getlayer(DualNetFull, 2, kernel_size=(5,5), padding='valid', name = 'DE_9_1')(xx)
    xx = DualNetFull.get_layer('DE_10_1')(xx)
    xx = CConv2D_getlayer(DualNetFull, 1, kernel_size=(5,5), padding='valid', name = 'DE_11')(xx)
    cos = DualNetFull.get_layer('DE_12')(xx)
    OneMatrix = DualNetFull.get_layer('DE_13')(cos)
    cos2 = DualNetFull.get_layer('DE_14')([cos,cos])
    sin_MAG = DualNetFull.get_layer('DE_15')([OneMatrix, cos2])
    sin_MAG = DualNetFull.get_layer('DE_16')(sin_MAG)
    sin = DualNetFull.get_layer('DE_17')([sin_MAG, Input2_DualNetPhase_Decoder])
    Ini_real = DualNetFull.get_layer('DE_18')([cos, Input3_DualNetPhase_Decoder])
    Ini_imag = DualNetFull.get_layer('DE_19')([sin, Input3_DualNetPhase_Decoder])
    Ini_est = DualNetFull.get_layer('DE_20')([Ini_real, Ini_imag])
    copyy = Ini_est
    xx = CConv2D_getlayer(DualNetFull, 16, kernel_size=(5,5), padding='valid', name = 'DE_21')(Ini_est)
    xx = DualNetFull.get_layer('DE_22')(xx)
    xx = CConv2D_getlayer(DualNetFull, 8, kernel_size=(5,5), padding='valid', name = 'DE_23')(xx)
    xx = DualNetFull.get_layer('DE_24')(xx)
    xx = CConv2D_getlayer(DualNetFull, 4, kernel_size=(5,5), padding='valid', name = 'DE_25')(xx)
    xx = DualNetFull.get_layer('DE_26')(xx)
    xx = CConv2D_getlayer(DualNetFull, 2, kernel_size=(5,5), padding='valid', name = 'DE_27')(xx)
    xx = DualNetFull.get_layer('DE_28')(xx)
    xx = DualNetFull.get_layer('DE_29')([xx,copyy])
    
    DualNetPhase_Decoder = Model(inputs=[Input1_DualNetPhase_Decoder,Input2_DualNetPhase_Decoder,Input3_DualNetPhase_Decoder,Input4_DualNetPhase_Decoder], outputs=[xx], name = 'DualNetPhase_Encoder')
    
    ################################### 
    adam = Adam(lr=0.001)
    DualNetMAG.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
    DualNetFull.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
    
    print('DualNetMAG')
    DualNetMAG.summary()
    if opt.from_scratch == 1:
        filepath='DualNetMAG'+'_CR_mag_'+str(opt.CR_MAG)+'_scen_'+opt.scen+'.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
    
        DualNetMAG.trainable = True
        DualNetMAG.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
        DualNetFull.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
        history1 = DualNetMAG.fit([x_train_DL, x_train_UL], y_train_DL,
                   batch_size=batch_size,
                   epochs=opt.n_epoch_mag,
                   validation_split=0.33,
                   callbacks = callbacks_list,
                   shuffle=True)
    
        f1 = 'DualNetMAG'+'_CR_mag_'+str(opt.CR_MAG)+'_scen_'+opt.scen+'.h5'
        DualNetMAG.load_weights(f1)
    
        
        filepath='DualNetFull'+'_CR_pha_'+str(opt.CR_PHA)+'_scen_'+opt.scen+'.h5'
        checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
        mode='min')
        callbacks_list2 = [checkpoint2]
        
        DualNetMAG.trainable = False
        DualNetMAG.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
        DualNetFull.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
    
        print('DualNetFull')
        DualNetFull.summary()
        #f1 = 'DualNetMAG'+'_QT'+str(opt.QT)+'_LD'+str(opt.ld)+'_loss'+str(opt.loss)+'_CR_pha'+str(opt.CR_PHA)+'.h5'
        #DualNetFull.load_weights(f1)
        history2 = DualNetFull.fit([x_train_DL, x_train_UL, x_train_DL_phase], y_train_DL_final,
                  batch_size=batch_size,
                  epochs = opt.n_epoch_pha,
                  validation_split=0.33,
                  callbacks = callbacks_list2,
                  shuffle=True)
        f1 = 'DualNetFull'+'_CR_pha_'+str(opt.CR_PHA)+'_scen_'+opt.scen+'.h5'
        DualNetFull.load_weights(f1)
    else:
        if opt.from_scratch_phase_only == 1:
            f1 = 'DualNetMAG'+'_CR_mag_'+str(opt.CR_MAG)+'_scen_'+opt.scen+'.h5'
            DualNetMAG.load_weights(f1)
            filepath='DualNetFull'+'_CR_pha_'+str(opt.CR_PHA)+'_scen_'+opt.scen+'.h5'
            checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
            mode='min')
            callbacks_list2 = [checkpoint2]
            
            DualNetMAG.trainable = False
            DualNetMAG.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
            DualNetFull.compile(loss=['mse'], optimizer=adam, metrics=['mse'])
            print('DualNetFull')
            DualNetFull.summary()
            history = DualNetFull.fit([x_train_DL, x_train_UL, x_train_DL_phase], y_train_DL_final,
                      batch_size=batch_size,
                      epochs = opt.n_epoch_pha,
                      validation_split=0.33,
                      callbacks = callbacks_list2,
                      shuffle=True)
            f1 = 'DualNetFull'+'_CR_pha_'+str(opt.CR_PHA)+'_scen_'+opt.scen+'.h5'
            DualNetFull.load_weights(f1)
        else:
            f1 = 'DualNetMAG'+'_CR_mag_'+str(opt.CR_MAG)+'_scen_'+opt.scen+'.h5'
            DualNetMAG.load_weights(f1)
            f1 = 'DualNetFull'+'_CR_pha_'+str(opt.CR_PHA)+'_scen_'+opt.scen+'.h5'
            DualNetFull.load_weights(f1)
#        f1 = 'DualNetMAG_pure_CCNN__outdoor3_CR4_new3.h5'
#        DualNetMAG.load_weights(f1)
#        f1 = 'DualNetMAG_SSQ_ResCCNN_k5_SMDP_outdoor3_CR4_8_full.h5'
#        DualNetFull.load_weights(f1)

x_test_DL = np.expand_dims(x_test_DL,-1)
x_test_UL = np.expand_dims(x_test_UL,-1)
y_test_DL = x_test_DL

tmp_complex = np.concatenate((x_test_DL,x_test_UL,y_test_DL),axis = 0)

x_test_DL = np.absolute(x_test_DL)
x_test_UL = np.absolute(x_test_UL)
y_test_DL = np.absolute(y_test_DL)

tmp = np.concatenate((x_test_DL,x_test_UL,y_test_DL),axis = 0)
tmp = np.reshape(tmp, (np.size(tmp,axis=0),width*height))
tmp = np.transpose(tmp)
tmp = min_max_scaler.fit_transform(tmp)
tmp = np.transpose(tmp)
tmp = np.reshape(tmp, (np.size(tmp,axis=0),width,height,1))

tmp_phase = np.angle(tmp_complex)
tmp_phase_cos = np.cos(tmp_phase)
tmp_real = np.multiply(tmp,np.cos(tmp_phase))
tmp_imag = np.multiply(tmp,np.sin(tmp_phase))

x_test_DL = tmp[0:np.size(x_test_DL,axis=0),:,:,0]
x_test_DL_phase_cos = tmp_phase_cos[0:np.size(x_test_DL,axis=0),:,:,0]
x_test_DL_phase = tmp_phase[0:np.size(x_test_DL,axis=0),:,:,0]

x_test_UL = tmp[np.size(x_test_UL,axis=0):2*np.size(x_test_UL,axis=0),:,:,0]
y_test_DL = tmp[2*np.size(y_test_DL,axis=0):3*np.size(y_test_DL,axis=0),:,:,0]
y_test_DL_phase_cos = tmp_phase_cos[2*np.size(y_test_DL,axis=0):3*np.size(y_test_DL,axis=0),:,:,0]
y_test_DL_phase = tmp_phase[2*np.size(y_test_DL,axis=0):3*np.size(y_test_DL,axis=0),:,:,0]
y_test_DL_real = tmp_real[2*np.size(y_test_DL,axis=0):3*np.size(y_test_DL,axis=0),:,:,0]
y_test_DL_imag = tmp_imag[2*np.size(y_test_DL,axis=0):3*np.size(y_test_DL,axis=0),:,:,0]

x_test_DL = np.expand_dims(x_test_DL,-1)
x_test_DL_phase_cos = np.expand_dims(x_test_DL_phase_cos,-1)
x_test_DL_phase = np.expand_dims(x_test_DL_phase,-1)
x_test_UL = np.expand_dims(x_test_UL,-1)
y_test_DL = np.expand_dims(y_test_DL,-1)
y_test_DL_real = np.expand_dims(y_test_DL_real,-1)
y_test_DL_imag = np.expand_dims(y_test_DL_imag,-1)
y_test_DL_phase_cos = np.expand_dims(y_test_DL_phase_cos,-1)
y_test_DL_phase = np.expand_dims(y_test_DL_phase,-1)

sipping_matrix = x_test_DL>=0.0
sipping_matrix = sipping_matrix.astype(float)

x_test_phase_cos = np.multiply(x_test_DL_phase_cos,sipping_matrix)
x_test_DL_phase_bit = x_test_DL_phase >= 0
x_test_DL_phase_bit = x_test_DL_phase_bit.astype(float)*2 - 1
x_test_DL_phase = np.concatenate((x_test_phase_cos,x_test_DL_phase_bit),axis = -1)
y_test_DL_final = np.concatenate((y_test_DL_real,y_test_DL_imag),axis = -1)

Est_MAG = DualNetMAG.predict([x_test_DL, x_test_UL])
[Est_codeword, Sign_code, Q_weighting] = DualNetPhase_Encoder.predict(x_test_DL_phase)
Est_codeword = np.multiply(Est_codeword, 2**(opt.QB-1))
Est_codeword = np.around(Est_codeword, decimals = 0)
Est_codeword = np.divide(Est_codeword, 2**(opt.QB-1))
tmp = Q_weighting[0:1,:]
Q_weighting = np.repeat(tmp,20000,0)
prediction0 = DualNetPhase_Decoder.predict([Est_codeword, Sign_code, Est_MAG, Q_weighting])
prediction = DualNetFull.predict([x_test_DL, x_test_UL, x_test_DL_phase])


NMSE_Final=np.zeros(1)
aaa=np.zeros(y_test_DL_final.shape[0])
if opt.scen == 'indoor':
    mat = loadmat('trunc_error_indoor.mat')
else:
    mat = loadmat('trunc_error_outdoor.mat')
residual_apl = mat['residual_apl'][0]
for k in range(y_test_DL_final.shape[0]):
    aaa[k]=20*np.math.log10((residual_apl[k]+np.linalg.norm(y_test_DL_final[k,:,:,0]-prediction0[k,:,:,0]+1j*(y_test_DL_final[k,:,:,1]-prediction0[k,:,:,1]),'fro'))/(residual_apl[k]+np.linalg.norm(y_test_DL_final[k,:,:,0]+1j*y_test_DL_final[k,:,:,1],'fro')))
NMSE_Final = np.mean(aaa)
