# Copyright (C) 2019 by Stefan Schubert
# https://www.tu-chemnitz.de/etit/proaut/en/team/stefanSchubert.html

# Project:
# https://www.tu-chemnitz.de/etit/proaut/ccnn

# If you use this source code in your work, please cite the following paper:
# Schubert, S., Neubert, P., PÃ¶schmann, J. & Protzel, P. (2019) Circular Convolutional Neural
# Networks for Panoramic Images and Laser Data. In Proc. of Intelligent Vehicles Symposium (IV)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.


# includes
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D


__version__ = 0.1  # version of the library


################################################################################
# Circular Convolutional Layer
def CConv2D(filters, kernel_size, name = '123',strides=(1, 1), activation='linear', padding='valid', kernel_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
    def CConv2D_inner(x):
        # padding (see https://www.tensorflow.org/api_guides/python/nn#Convolution)
        in_height = int(x.get_shape()[1])
        in_width = int(x.get_shape()[2])

        if (in_height % strides[0] == 0):
            pad_along_height = max(kernel_size[0] - strides[0], 0)
        else:
            pad_along_height = max(
                kernel_size[0] - (in_height % strides[0]), 0)
        if (in_width % strides[1] == 0):
            pad_along_width = max(kernel_size[1] - strides[1], 0)
        else:
            pad_along_width = max(kernel_size[1] - (in_width % strides[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # left and right side for padding
        pad_left = Cropping2D(cropping=((0, 0), (in_width-pad_left, 0)))(x)
        pad_right = Cropping2D(cropping=((0, 0), (0, in_width-pad_right)))(x)

        # add padding to incoming image
        conc = Concatenate(axis=2)([pad_left, x, pad_right])
        
        # left and right side for padding
        pad_top = Cropping2D(cropping=((in_height-pad_top, 0),(0, 0)))(conc)
        pad_down = Cropping2D(cropping=((0, in_height-pad_bottom),(0, 0)))(conc)
        
        # add padding to incoming image
        conc = Concatenate(axis=1)([pad_top, conc, pad_down])
        # top/bottom padding options
        if padding == 'same':
            conc = ZeroPadding2D(padding={'top_pad': pad_top,
                                          'bottom_pad': pad_bottom})(conc)
        elif padding == 'valid':
            pass
        else:
            raise Exception('Padding "{}" does not exist!'.format(padding))

        # perform the circular convolution
        if name == '123':
            cconv2d = Conv2D(filters=filters, kernel_size=kernel_size,
                             strides=strides, activation=activation,
                             padding='valid',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)(conc)
        else:
            cconv2d = Conv2D(filters=filters, kernel_size=kernel_size,
                             strides=strides, activation=activation,
                             padding='valid',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer, name=name)(conc)

        # return circular convolution layer
        return cconv2d
    return CConv2D_inner

def CConv2D_getlayer(Model, filters, kernel_size, name, strides=(1, 1), activation='linear', padding='valid', kernel_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
    def CConv2D_inner_getlayer(x):
        # padding (see https://www.tensorflow.org/api_guides/python/nn#Convolution)
        in_height = int(x.get_shape()[1])
        in_width = int(x.get_shape()[2])

        if (in_height % strides[0] == 0):
            pad_along_height = max(kernel_size[0] - strides[0], 0)
        else:
            pad_along_height = max(
                kernel_size[0] - (in_height % strides[0]), 0)
        if (in_width % strides[1] == 0):
            pad_along_width = max(kernel_size[1] - strides[1], 0)
        else:
            pad_along_width = max(kernel_size[1] - (in_width % strides[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # left and right side for padding
        pad_left = Cropping2D(cropping=((0, 0), (in_width-pad_left, 0)))(x)
        pad_right = Cropping2D(cropping=((0, 0), (0, in_width-pad_right)))(x)

        # add padding to incoming image
        conc = Concatenate(axis=2)([pad_left, x, pad_right])
        
        # left and right side for padding
        pad_top = Cropping2D(cropping=((in_height-pad_top, 0),(0, 0)))(conc)
        pad_down = Cropping2D(cropping=((0, in_height-pad_bottom),(0, 0)))(conc)
        
        # add padding to incoming image
        conc = Concatenate(axis=1)([pad_top, conc, pad_down])
        # top/bottom padding options
        if padding == 'same':
            conc = ZeroPadding2D(padding={'top_pad': pad_top,
                                          'bottom_pad': pad_bottom})(conc)
        elif padding == 'valid':
            pass
        else:
            raise Exception('Padding "{}" does not exist!'.format(padding))

        # perform the circular convolution
        cconv2d = Model.get_layer(name)(conc)

        # return circular convolution layer
        return cconv2d
    return CConv2D_inner_getlayer
################################################################################
# Circular Transposed Convolutional Layer (Circular Deconvolutional Layer)
def CConv2DTranspose(filters, kernel_size, strides=(1, 1), activation='linear', padding='valid', kernel_initializer='glorot_uniform', kernel_regularizer=None):
    def CConv2DTranspose_inner(x):
        # width of incoming image
        x_width = int(x.get_shape()[2])

        # determine required addtional attachment and cropping width
        pad_width = int(
            0.5 + (kernel_size[1] - 1.) / (2. * strides[1]))  # ceil
        crop = pad_width * strides[1]

        # left and right side for padding
        pad_left = Cropping2D(cropping=((0, 0), (x_width-pad_width, 0)))(x)
        pad_right = Cropping2D(cropping=((0, 0), (0, x_width-pad_width)))(x)

        # add padding to incoming image
        conc = Concatenate(axis=2)([pad_left, x, pad_right])

        # top/bottom padding options
        if padding == 'same':
            pass
        elif padding == 'valid':  # TODO
            raise Exception('Valid padding has not yet been implemented.')
        else:
            raise Exception('Padding "{}" does not exist!'.format(padding))

        # perform the circular convolution
        cconv2dtranspose = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                           strides=strides, activation=activation,
                                           padding='same',
                                           kernel_initializer=kernel_initializer,
                                           kernel_regularizer=kernel_regularizer)(conc)

        # crop it to the same shape (multiplied by stride)
        croped = Cropping2D(cropping=((0, 0), (crop, crop)))(cconv2dtranspose)

        # return circular convolution layer
        return croped
    return CConv2DTranspose_inner


################################################################################
# run all layer types and compare to their linear counterparts
if __name__ == '__main__':
    print('''##########################################################################################
These are the examples from the websiteÂ´s animations (www.tu-chemnitz.de/etit/proaut/ccnn)
##########################################################################################\n''')

    from keras.layers import Input
    import numpy as np

    # create model for CConv2D
    input1 = Input((1, 6, 1))
    output = CConv2D(1, (1, 3), strides=(1, 1),
                     activation='linear', padding='same')(input1)
    m_cconv = keras.models.Model(input1, output)

    w = m_cconv.get_weights()
    w[0][0, :, 0, 0] = [1, 0, -1]
    m_cconv.set_weights(w)

    # create model for Conv2DTranspose
    input1 = Input((1, 6, 1))
    output = Conv2D(1, (1, 3), strides=(1, 1),
                    activation='linear', padding='same')(input1)
    m_conv = keras.models.Model(input1, output)

    m_conv.set_weights(w)

    # run linear and circular transposed convolution
    x = np.zeros((1, 1, 6, 1))
    x[0, 0, :, 0] = [1, 2, 1, 0, 2, 1]

    # output results from convolutions
    print('\ninput vector:\n', x)
    print('circular convolution:\n', m_cconv.predict(x))
    print('linear convolution:\n', m_conv.predict(x))

    # create model for CConv2DTranspose
    input1 = Input((1, 3, 1))
    output = CConv2DTranspose(1, (1, 3), strides=(1, 2),
                              activation='linear', padding='same')(input1)
    m_cconvt = keras.models.Model(input1, output)

    w = m_cconvt.get_weights()
    w[0][0, :, 0, 0] = [-1, 2, 1]
    m_cconvt.set_weights(w)

    # create model for Conv2DTranspose
    input1 = Input((1, 3, 1))
    output = Conv2DTranspose(1, (1, 3), strides=(1, 2),
                             activation='linear', padding='same')(input1)
    m_convt = keras.models.Model(input1, output)

    m_convt.set_weights(w)

    # run linear and circular transposed convolution
    x = np.zeros((1, 1, 3, 1))
    x[0, 0, :, 0] = [1, 2, 3]

    # output results from transposed convolutions
    print('\ninput vector:\n', x)
    print('circular transposed convolution:\n', m_cconvt.predict(x))
    print('linear transposed convolution:\n', m_convt.predict(x))