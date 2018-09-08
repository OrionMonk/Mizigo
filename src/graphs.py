import tensorflow as tf 

def assertRaise(condition, error, msg):
    try:
        assert condition 
    except:
        raise error(msg)

class BaseGraph():
    def __init_ ( self ):
        pass

class VanillaGraph(BaseGraph):
    def __init__(self, D, D_in, G, G_in, G_scope, D_scope):
        
        # generate fake samples
        with tf.variable_scope( G_scope) as scope:
            self.G_out = G.generate ( G_in, D_in.shape[1:] )
            
        # verify generator output
        assertRaise ( isinstance ( self.G_out, tf.Tensor ), TypeError, "Generator should output a Tensorflow Tensor Object" )
        assertRaise ( self.G_out.shape.as_list() == D_in.shape.as_list(), ValueError, "Generator Output and Sample Input Shape mismatch!\nGenerator : "+','.join(str(x) for x in self.G_out.shape.as_list())+"\nDiscriminator : "+','.join(str(x) for x in D_in.shape.as_list()))
        
        # discrimination of real and fake samples
        with tf.variable_scope( D_scope ) as scope:
            self.D_out_real = D.discriminate ( D_in )

        with tf.variable_scope( D_scope ) as scope:
            scope.reuse_variables()
            self.D_out_fake = D.discriminate ( self.G_out )

        # Verify discriminator output
        msg = "Discriminator should output a Tensorflow Tensor Object"
        assertRaise ( isinstance ( self.D_out_real, tf.Tensor ) and isinstance ( self.D_out_fake, tf.Tensor ), TypeError, msg )

        msg = "Discriminator Output expected shape : (None, 1). Got : " + ', '.join ( str ( self.D_out_real.shape ) )
        assertRaise( len ( self.D_out_real.shape ) == 2 and len ( self.D_out_fake.shape ) == 2 and self.D_out_real.shape[1] == 1 and self.D_out_fake.shape[1] == 1, ValueError, msg )
        