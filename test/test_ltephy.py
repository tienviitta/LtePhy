#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import *
from ltephy.ofdmsym.ofdmsym import OfdmSym
import numpy as np

phy_res_blocks = [6, 15, 25, 50, 75, 100]
subcarriers = 12

def setup():
    pass
def teardown():
    pass

def test_ofdmsym_sys_bw0():
    '''
    Test:OfdmSym:test_ofdmsym_sys_bw0
    '''
    # Parameters
    params = {
        'sys_bw': 0,
        'normal_cp': True
    }
    # OFDM symbol generator
    osg = OfdmSym(params)
    # Input symbols
    input_size = (14 if params['normal_cp'] else 12, subcarriers * phy_res_blocks[params['sys_bw']])
    syms = np.sqrt(1.0/2.0) * \
        ((2 * np.random.randint(2, size=input_size) - 1) + \
        1j*(2 * np.random.randint(2, size=input_size) - 1))
    # Chain
    tx = osg.tx(syms)
    rx = osg.rx(tx, syms)
    # Test(s)
    np.testing.assert_allclose(rx, syms)
