from nose.tools import *
from ltephy.ofdmsym.ofdmsym import OfdmSym
import numpy as np

def setup():
    pass
def teardown():
    pass

def test_ofdmsym_tx_rx_real():
    '''
    Test:OfdmSym:test_ofdmsym_tx_rx_real
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

def test_ofdmsym_tx_rx_complex():
    '''
    Test:OfdmSym:test_ofdmsym_tx_rx_complex
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
