import tensorflow as tf 
import numpy as np

def assertRaise(condition, error, msg):
    try:
        assert condition 
    except:
        raise error(msg)

class Generator ():
    def generate ( G_in ):
        return None

class Generator1D ( Generator ):
    def generate ( self, G_in, out_shape ):
        return tf.layers.dense ( G_in, units = out_shape[0] )

class Generator2D ( Generator ):
    # format output to desired shape
    def format_output ( self, flow, shape ):
        return tf.reshape ( flow, [-1] + list( shape ) )

    def generate ( self, G_in, out_shape ):
        flow = G_in
        flow = tf.layers.dense ( flow, units = np.prod ( out_shape ) // 2 )
        flow = tf.nn.elu ( flow )
        flow = tf.layers.dense ( flow, units = np.prod ( out_shape) )
        flow = tf.nn.relu ( flow )
        return self.format_output ( flow, out_shape )

class Generator2DCNN ( Generator2D ):
    def generate ( self, G_in, out_shape ):
        flow = G_in
        assertRaise ( out_shape[0] >= 16 and out_shape[1] >= 16 , ValueError, "Images cannot have height or width smaller than 16 pixels")
        
        h = out_shape[0] - 4 * 2
        w = out_shape[1] - 4 * 2
        channels = 64
        
        flow = tf.layers.dense ( flow, units = h * w * channels )
        flow = tf.nn.elu ( flow )
        
        flow = tf.layers.dense ( flow, units = h * w * channels )
        flow = tf.nn.elu ( flow )

        flow = tf.reshape ( flow, [-1, h, w, channels ] )

        flow = tf.layers.conv2d_transpose ( flow, filters = 32, kernel_size = 5, strides = 1, padding = 'valid' )
        flow = tf.nn.elu ( flow )

        flow = tf.layers.conv2d_transpose ( flow, filters = 3, kernel_size = 5, strides = 1, padding = 'valid' )
        flow = tf.nn.relu ( flow )
        
        return self.format_output ( flow, out_shape )