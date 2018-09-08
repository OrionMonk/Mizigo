import tensorflow as tf

def assertRaise(condition, error, msg):
    try:
        assert condition 
    except:
        raise error(msg)

class IteratedTrainer():
    def _shuffle ( self, data ):
        assertRaise ( isinstance ( data, np.ndarray ), TypeError, 'data : expects a numpy array')
        permute = np.random.permutation ( data.shape[0] )
        return data[permute]

    def run ( self, core = [], metrics = [], metric_names = [], save_path = None, feed_iterator = {}, epochs = 10, callbacks = [] ):
        assertRaise ( isinstance ( core, list ), TypeError, "core need to be a list" )
        assertRaise ( all ( isinstance ( item, tf.Operation ) for item in core ), TypeError, "core need to be a list of Tensorflow Tensor Objects" )
        
        assertRaise ( isinstance ( metrics, list ), TypeError, "metrics need to be a list" )
        assertRaise ( all ( isinstance ( item, tf.Tensor ) for item in metrics ), TypeError, "metrics need to be a list of Tensorflow Tensor Objects" )
        
        assertRaise ( isinstance ( metric_names, list ), TypeError, "metric_names need to be a list" )
        assertRaise ( all ( isinstance ( item, str ) for item in metric_names ), TypeError, "metric_names need to be a list of Tensorflow Tensor Objects" )
        
        assertRaise ( len ( metrics ) == len ( metric_names ), ValueError, "metrics and metric_names have to be of same length" )

        assertRaise ( isinstance ( feed_iterator, list   ), TypeError, "feed_iterator needs to be a dictionary" )
        assertRaise ( all ( isinstance ( feed_iterator[i][0], tf.Tensor ) for i in range(len(feed_iterator)) ), TypeError, "feed_iterator needs to be a dictionary of tensorflow.Tensor objects as keys and iterators as values" )
        

        # Get tensorflow model saver
        if save_path != None:
            assertRaise ( isinstance ( save_path, str ), TypeError, "Save path has to a string") 
            saver = tf.train.Saver ()

        with tf.Session () as sess:
            # Load saved model if exists
            if save_path != None:
                try:
                    saver.restore ( sess, save_path )
                    print( "Loaded saved model named ", save_path )
                except:
                    print( "No saved model found in save path. Initializing Global Variables. " )
                    sess.run( tf.global_variables_initializer () )
            else:
                sess.run( tf.global_variables_initializer () )

            for epoch in range ( epochs ):
                logs = None
                feed_dict_copy = [ ( feed_iterator[i][0], feed_iterator[i][1].duplicate() ) for i in range(len(feed_iterator)) ]
                
                while True:
                    try:
                        feed_dict = { feed_dict_copy[i][0] : next ( feed_dict_copy[i][1] ) for i in range(len(feed_dict_copy)) }
                        sess.run( core, feed_dict )
                        logs = sess.run ( metrics, feed_dict )

                        if save_path != None:
                            saver.save ( sess, save_path )
                    # epoch ends
                    except StopIteration:
                        # means epoch has atleast one batch run
                        if logs != None:
                            for call in callbacks:
                                call( epoch, { metric_names[i] : logs[i] for i in range ( len ( logs ) ) } )
                        break
                        
        