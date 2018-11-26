
import numpy as np
import torch

import models.base

class Model(models.base.ModelBase):

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

    def _get_generator(self, z_shape, image_shape):

        gf_dim = 64
        s16 = (int(image_shape[1]/16), int(image_shape[2]/16))

        net = models.base.SequentialContainer(z_shape)

        net.add_Dense(gf_dim*8*s16[0]*s16[1])
        net.add_Reshape((gf_dim//2, 4*s16[0], 4*s16[1]))
        net.add_Activation(self.g_act)

        net.add_Conv2DTranspose(gf_dim, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)), output_padding=(1,1))
        net.add_Activation(self.g_act)

        net.add_Conv2DTranspose(gf_dim, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)), output_padding=(1,1))
        net.add_Activation(self.g_act)

        net.add_Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding = models.base.get_same_padding((3,3)))

        if self.g_tanh:
           net.add_Activation(torch.nn.Tanh())

        return net.get()

    def _get_discriminator(self, image_shape):
    
        df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    
        net = models.base.SequentialContainer(image_shape)

        net.add_Conv2D(df_dim, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)

        net.add_Conv2D(df_dim*2, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)

        net.add_Conv2D(df_dim*4, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)

        net.add_Conv2D(df_dim*8, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)
    
        net.add_Flatten()
        net.add_Dense(1)
            
        return net.get()

class ResidualModel(models.base.ModelBase):

    def __init__(self, **kwargs):
        super(ResidualModel, self).__init__(**kwargs)

    def _get_generator(self, z_shape, image_shape):

        s16 = (int(image_shape[1]/16), int(image_shape[2]/16))
        gf_dim = 64 # Dimension of gen filters in first conv layer. [64]

        net = models.base.SequentialContainer(z_shape)

        net.add_Dense(gf_dim*8*s16[0]*s16[1])
        net.add_Reshape((gf_dim*8, s16[0], s16[1]))
        net.add_Activation(self.g_act)

        net.add_Residual(gf_dim*8, kernel_size=(3, 3), activation=self.g_act)
        #net.add_Conv2D(gf_dim*8, kernel_size=(3, 3), padding = models.base.get_same_padding((3,3)))
        net.add_Conv2DTranspose(gf_dim*8, kernel_size=(3, 3), strides=(2,2), padding = models.base.get_same_padding((3,3)), output_padding=(1,1))
        net.add_Activation(self.g_act)

        net.add_Residual(gf_dim*4, kernel_size=(3, 3), activation=self.g_act)
        #net.add_Conv2D(gf_dim*4, kernel_size=(3, 3), padding = models.base.get_same_padding((3,3)))
        net.add_Conv2DTranspose(gf_dim*4, kernel_size=(3, 3), strides=(2,2), padding = models.base.get_same_padding((3,3)), output_padding=(1,1))
        net.add_Activation(self.g_act)

        net.add_Residual(gf_dim*2, kernel_size=(3, 3), activation=self.g_act)
        #net.add_Conv2D(gf_dim*2, kernel_size=(3, 3), padding = models.base.get_same_padding((3,3)))
        net.add_Conv2DTranspose(gf_dim*2, kernel_size=(3, 3), strides=(2,2), padding = models.base.get_same_padding((3,3)), output_padding=(1,1))
        net.add_Activation(self.g_act)

        net.add_Residual(gf_dim, kernel_size=(3, 3), activation=self.g_act)
        #net.add_Conv2D(gf_dim, kernel_size=(3, 3), padding = models.base.get_same_padding((3,3)))
        net.add_Conv2DTranspose(gf_dim, kernel_size=(3, 3), strides=(2,2), padding = models.base.get_same_padding((3,3)), output_padding=(1,1))
        net.add_Activation(self.g_act)

        net.add_Residual(16, kernel_size=(3, 3), activation=self.g_act)
        #net.add_Conv2D(16, kernel_size=(3, 3), padding = models.base.get_same_padding((3,3)))
        net.add_Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding = models.base.get_same_padding((3,3)))

        if self.g_tanh:
           net.add_Activation(torch.nn.Tanh())

        return net.get()

    def _get_discriminator(self, image_shape):
    
        df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    
        net = models.base.SequentialContainer(image_shape)

        net.add_Conv2D(df_dim, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)

        net.add_Residual(df_dim, kernel_size=(3, 3), activation=self.d_act)

        net.add_Conv2D(df_dim*2, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)

        net.add_Residual(df_dim*2, kernel_size=(3, 3), activation=self.d_act)

        net.add_Conv2D(df_dim*4, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)

        net.add_Residual(df_dim*4, kernel_size=(3, 3), activation=self.d_act)

        net.add_Conv2D(df_dim*8, kernel_size=(5, 5), strides=(2,2), padding = models.base.get_same_padding((5,5)))
        net.add_Activation(self.d_act)
    
        net.add_Flatten()
        net.add_Dense(1)
            
        return net.get()

class MLPModel(models.base.ModelBase):
    def __init__(self, **kwargs):
        super(MLPModel, self).__init__(**kwargs)

    def _get_generator(self, z_shape, image_shape):

        s16 = (int(image_shape[1]/16), int(image_shape[2]/16))
        gf_dim = 64 # Dimension of gen filters in first conv layer. [64]

        net = models.base.SequentialContainer(z_shape)

        net.add_Dense(256)
        net.add_Activation(self.g_act)

        net.add_Dense(512)
        net.add_Activation(self.g_act)

        net.add_Dense(1024)
        net.add_Activation(self.g_act)

        net.add_Dense(2056)
        net.add_Activation(self.g_act)

        net.add_Dense(3*s16[0]*16*s16[1]*16)
        net.add_Reshape((3, image_shape[1], image_shape[2]))

        if self.g_tanh:
           net.add_Activation(torch.nn.Tanh())

        return net.get()

    def _get_discriminator(self, image_shape):
    
        df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    
        net = models.base.SequentialContainer(image_shape)

        net.add_Reshape((3*image_shape[1]*image_shape[2],))

        net.add_Dense(1024)
        net.add_Activation(self.d_act)

        net.add_Dense(512)
        net.add_Activation(self.d_act)

        net.add_Dense(256)
        net.add_Activation(self.d_act)

        net.add_Dense(128)
        net.add_Activation(self.d_act)

        net.add_Dense(1)
        return net.get()