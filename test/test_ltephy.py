from nose.tools import *
from ltephy.ofdmsym.ofdmsym import OfdmSym
import numpy as np

def setup():
    pass
def teardown():
    pass

def test_ofdmsym():
    '''
    Test:OfdmSym:tx and rx
    '''
    params = {
        'nFFT': 128
    }
    osg = OfdmSym(params)
    syms = 2 * np.random.randint(2, size=params['nFFT']) - 1
    tx = osg.tx(syms)
    rx = osg.rx(tx)
    np.testing.assert_array_almost_equal(rx, syms, decimal=14)
