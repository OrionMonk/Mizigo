import tensorflow as tf 
import numpy as np 

class Discriminator ():
    def discriminate (self, D_in):
        return None

class Discriminator1D ( Discriminator ):
    def discriminate ( self, D_in ):
        return tf.layers.dense(D_in, units = 1)

class Discriminator2D ( Discriminator ):
    def discriminate ( self, D_in ):
        flow = tf.contrib.layers.flatten ( D_in )
        flow = tf.layers.dense ( flow, units = np.prod ( D_in.shape )//2 )
        flow = tf.nn.elu ( flow )
        flow = tf.layers.dense ( flow, units = 1 )
        flow = tf.nn.elu ( flow )
        return flow

class Discriminator2DCNN ( Discriminator ):
    def discriminate ( self, D_in ):
        flow = D_in 
        
        flow = tf.layers.conv2d ( flow, filters = 32, kernel_size = 5, padding = 'same' )
        flow = tf.nn.elu ( flow )
        flow = tf.layers.conv2d ( flow, filters = 64, kernel_size = 5, padding = 'same' )
        flow = tf.nn.elu ( flow )
        
        flow = tf.layers.max_pooling2d ( flow, pool_size = 2, strides = 2 )

        flow = tf.contrib.layers.flatten ( flow )
        flow = tf.layers.dense ( flow, units = 1 )
        flow = tf.nn.elu ( flow )

        return flow