import numpy as np
import glob 
import cv2, copy
from random import shuffle

def assertRaise(condition, error, msg):
    try:
        assert condition 
    except:
        raise error(msg)

class Iterator():
    def __next__ ( self ):
        return None
    def get_shape ( self ):
        return None

class NoiseIterator ( Iterator ):
    def __init__ ( self, size):
        msg = 'size : expects python array-list. Got type'+str ( type ( size ) )
        assertRaise ( isinstance ( size, list ), TypeError, msg)
        self.size = size
     
    def duplicate ( self ):
        return copy.copy ( self )

    def __next__ ( self ):
        return np.random.uniform(-1., 1., size = self.size)

    def get_shape ( self ):
        return self.size

class BatchedFileIterator ( Iterator ):
    def __init__ ( self, files, target_shape, batch_size = 32 ):
        assertRaise ( isinstance ( target_shape, tuple ) and len ( target_shape ) == 2, ValueError, "Invalid Target Shape for image! Expecting tuple of size 2" )
        
        # need to shuffle files first
        self.files = glob.glob ( files )
        self.index = 0
        self.batch_size = batch_size
        self.target_shape = target_shape

    def duplicate ( self ):
        shuffle ( self.files )
        return copy.copy(self)
    
    def load ( self, file_name ):
        return None

    def __next__ ( self ):
        files = []
        count = 0
        while self.index < len ( self.files ) and count < self.batch_size:
            file_data = self.load ( self.files [ self.index ] )
            files.append ( file_data )
            count += 1
            self.index += 1
        
        result = np.array ( files )

        try:
            assert result.shape[0] == self.batch_size
        except:
            raise StopIteration

        return result

    def get_shape ( self ):
        return list ( self.target_shape )

class BatchedImageIterator ( BatchedFileIterator ):
    def get_shape ( self ):
        return list ( self.target_shape ) + [3]

    def load( self, file_name ):
        return np.array ( cv2.cvtColor( cv2.resize ( cv2.imread ( file_name ), self.target_shape ), cv2.COLOR_BGR2RGB) ) / 255.0 