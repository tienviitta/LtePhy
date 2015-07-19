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
        syms_in_shifted = np.fft.fftshift(syms_in, axes=0)
        syms_fd = np.fft.fft(syms_in_shifted, axis=0)
        syms_td = np.fft.ifft(syms_fd, axis=0)
        return syms_td

    def rx(self, syms_in):
        '''
        OfdmSym:rx
        '''
        syms_fd = np.fft.fft(syms_in, axis=0)
        syms_td = np.fft.ifft(syms_fd, axis=0)
        syms_out_shifted = np.fft.ifftshift(syms_td, axes=0)
        return syms_out_shifted

