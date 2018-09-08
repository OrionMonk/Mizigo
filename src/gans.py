import tensorflow as tf
import numpy as np
from . import graphs, generators, discriminators, trainers, iterators
import inspect

def assertRaise(condition, error, msg):
    try:
        assert condition 
    except:
        raise error(msg)

class Gan ():
    def __init__ ( self, generator = None, discriminator = None, graph_model = None, noise_dim = 10, optimizer = None):
        self._G = generator 
        self._D = discriminator
        self._graph_model = graph_model
        self._noise_dim = noise_dim
        self._optimizer = optimizer
        self._D_scope = 'scope_dis'
        self._G_scope = 'scope_gen'

    def _split_to_batches ( self, data, batches ):
        assertRaise ( isinstance ( data, np.ndarray ), TypeError, 'data : expects a numpy array')
        return np.array ( np.split ( data, batches ) )

    def _get_D_loss ( self, D_out_fake, D_out_real):
        return tf.reduce_mean ( D_out_fake ) - tf.reduce_mean ( D_out_real )

    def _get_G_loss ( self, D_out_fake ):
        return - tf.reduce_mean ( D_out_fake )

    def train ( self, data_iterator, epochs = 10, batch_size = 32, lr = 0.01, save_path = None, callbacks = [] ):
        assertRaise ( isinstance ( self._G, generators.Generator ), TypeError, "Generator parameter has to be a generator object" )
        assertRaise ( isinstance ( self._D, discriminators.Discriminator ), TypeError, "Discriminator Parameter is not a discriminator object")
        assertRaise ( isinstance ( data_iterator, iterators.Iterator ), TypeError, "Data Samples are supposed to be an iterator" )
        assertRaise ( isinstance ( self._noise_dim, int ) and self._noise_dim > 0, ValueError, "Noise dimension has to be a positive integer"  )
        assertRaise ( isinstance ( epochs, int ) and epochs > 0, ValueError, "Epochs has to be a positive integer")
        assertRaise ( isinstance ( batch_size, int ) and batch_size > 0, TypeError, "Batch size has to be a positive integer" )
        assertRaise ( ( isinstance ( lr, float ) or isinstance ( lr, int) ) and lr > 0, TypeError, "Learning Rate expects a positive float value." )
        assertRaise ( isinstance ( self._optimizer, tf.train.Optimizer ) ,TypeError, "Optimizer needs to be a Tensorflow Optimizer Object" )
        
        self._D_in_real = tf.placeholder ( tf.float32, shape = [ batch_size ] + data_iterator.get_shape() )
        self._G_in = tf.placeholder ( tf.float32, shape = [batch_size, self._noise_dim] )
        
        # connect generator and discriminator 
        self._graph = self._graph_model( self._D, self._D_in_real, self._G, self._G_in, self._G_scope, self._D_scope )

        # losses
        self._D_loss = self._get_D_loss ( self._graph.D_out_fake, self._graph.D_out_real )
        self._G_loss = self._get_G_loss ( self._graph.D_out_fake )

        # generator _optimizer
        theta_d = [ var for var in tf.trainable_variables () if self._D_scope in var.name ]
        self.discriminator_optimizer = self._optimizer.minimize ( self._D_loss, var_list = theta_d )

        # discriminator _optimizer
        theta_g = [ var for var in tf.trainable_variables () if self._G_scope in var.name ]
        self.generator_optimizer = self._optimizer.minimize ( self._G_loss, var_list = theta_g )


        # get random batched data_iterator
        # data = self._split_to_batches( self._shuffle( data_iterator ), batches )

        trainers.IteratedTrainer ().run (
            core = [ self.generator_optimizer, self.discriminator_optimizer ],
            metrics = [ self._G_loss, self._D_loss,  self._graph.G_out],
            metric_names = ['Generator loss', 'Discriminator loss', 'Generated samples'],
            feed_iterator = [
                ( self._D_in_real , data_iterator ),
                ( self._G_in , iterators.NoiseIterator ( size =  list ( self._G_in.shape ) ) )
            ],
            epochs = epochs,
            save_path = save_path,
            callbacks = callbacks 
        )


class Gan1D ( Gan ):
    def __init__ ( self, noise_dim = 10 ,lr = 0.001 ):
        super ().__init__ ( 
            generators.Generator1D (), 
            discriminators.Discriminator1D (), 
            graphs.VanillaGraph, 
            noise_dim, 
            tf.train.AdamOptimizer(lr) 
        )

class Gan2D ( Gan ):
    def __init__ (self, noise_dim = 10, lr = 0.001):
        super().__init__ ( 
            generators.Generator2D (), 
            discriminators.Discriminator2D (), 
            graphs.VanillaGraph, 
            noise_dim, 
            tf.train.AdamOptimizer(lr) 
        )

class Gan2DCNN ( Gan ):
     def __init__ (self, noise_dim = 10, lr = 0.001):
        super().__init__ ( 
            generators.Generator2DCNN (), 
            discriminators.Discriminator2DCNN (), 
            graphs.VanillaGraph, 
            noise_dim, 
            tf.train.AdamOptimizer(lr) 
        )
