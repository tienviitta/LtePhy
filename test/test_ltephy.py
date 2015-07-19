from nose.tools import *
from ltephy.ofdmsym.ofdmsym import OfdmSym
import numpy as np

def setup():
    pass
def teardown():
    pass

def test_ofdmsym_real_vec():
    '''
    Test:OfdmSym:test_ofdmsym_real_vec
    '''
    # Parameters
    params = {
        'nFFT': 128
    }
    # OFDM symbol generator
    osg = OfdmSym(params)
    # Input symbols
    syms = 2 * np.random.randint(2, size=params['nFFT']) - 1
    # Chain
    tx = osg.tx(syms)
    rx = osg.rx(tx)
    # Test(s)
    np.testing.assert_allclose(rx, syms)
    #np.testing.assert_array_almost_equal(rx, syms, decimal=14)

def test_ofdmsym_complex_vec():
    '''
    Test:OfdmSym:test_ofdmsym_complex_vec
    '''
    # Parameters
    params = {
        'nFFT': 2048
    }
    # OFDM symbol generator
    osg = OfdmSym(params)
    # Input symbols
    syms = (2 * np.random.randint(2, size=params['nFFT']) - 1) + 1j*(2 * np.random.randint(2, size=params['nFFT']) - 1)
    # Chain
    tx = osg.tx(syms)
    rx = osg.rx(tx)
    # Test(s)
    np.testing.assert_allclose(rx, syms)
    #np.testing.assert_array_almost_equal(rx, syms, decimal=14)

def test_ofdmsym_complex_mat():
    '''
    Test:OfdmSym:test_ofdmsym_complex_mat
    '''
    # Parameters
    params = {
        'nFFT': 2048,
        'nSym': 14
    }
    # OFDM symbol generator
    osg = OfdmSym(params)
    # Input symbols
    syms = \
        (2 * np.random.randint(2, size=(params['nFFT'],params['nSym'])) - 1) + \
        1j*(2 * np.random.randint(2, size=(params['nFFT'],params['nSym'])) - 1)
    # Chain
    tx = osg.tx(syms)
    rx = osg.rx(tx)
    # Test(s)
    np.testing.assert_allclose(rx, syms)
    #np.testing.assert_array_almost_equal(rx, syms, decimal=14)
