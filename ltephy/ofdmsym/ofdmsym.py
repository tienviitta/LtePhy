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
        return np.fft.fft(syms_in)

    def rx(self, syms_in):
        '''
        OfdmSym:rx
        '''
        return np.fft.ifft(syms_in)

