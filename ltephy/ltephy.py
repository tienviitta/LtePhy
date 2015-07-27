#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
LtePhy:module
'''

class LtePhy(object):
    '''
    LtePhy:class
    '''

    def __init__(self, params):
        '''
        :param params:
        '''
        # LTE parameters
        self.subcarriers = 12
        self.phy_res_blocks = [6, 15, 25, 50, 75, 100]
        self.fft_sizes = [128, 256, 512, 1024, 1536, 2048]
        self.bw_factors = [16, 8, 4, 2, 4.0/3.0, 1]
        # LTE DL derived parameters
        self.n_fft = self.fft_sizes[params['sys_bw']]
        self.normal_cp = np.array(2 * [160, 144, 144, 144, 144, 144, 144]) / self.bw_factors[params['sys_bw']]
        self.extended_cp = np.array(2 * [512, 512, 512, 512, 512, 512]) / self.bw_factors[params['sys_bw']]
        self.n_prb = self.phy_res_blocks[params['sys_bw']]
        self.n_subc = self.subcarriers * self.n_prb
        self.n_ofdm_sym = 14 if params['normal_cp'] else 12
        self.cp = self.normal_cp if params['normal_cp'] else self.extended_cp
        self.n_sym = self.cp.sum() + (self.n_ofdm_sym * self.n_fft)
        self.n_gb = self.n_fft - self.n_subc - 1
        # LTE DL OFDM symbol start and stop indexes
        self.tx_sym_start_indexes = np.cumsum(np.hstack([0, self.cp[:-1] + self.n_fft]))
        self.tx_sym_stop_indexes = np.cumsum(np.hstack([self.cp + self.n_fft]))
        self.rx_sym_start_indexes = self.cp + self.tx_sym_start_indexes
        self.rx_sym_stop_indexes = self.rx_sym_start_indexes + self.n_fft
