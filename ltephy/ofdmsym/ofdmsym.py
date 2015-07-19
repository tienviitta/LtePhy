'''
OfdmSym:module
'''

import numpy as np

class OfdmSym(object):
    
    '''
    OfdmSym:class
    '''

    def __init__(self, params):
        '''
        OfdmSym:__init__
        '''
        super(OfdmSym, self).__init__()
        self.params = params

    def tx(self, syms_in):
        '''
        OfdmSym:tx
        '''
        return np.fft.ifft(np.fft.fft(np.fft.fftshift(syms_in, axes=0), axis=0), axis=0)

    def rx(self, syms_in):
        '''
        OfdmSym:rx
        '''
        return np.fft.ifftshift(np.fft.ifft(np.fft.fft(syms_in, axis=0), axis=0), axes=0)

