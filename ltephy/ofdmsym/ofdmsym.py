#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LTE DL OFDM Symbol Processing:module
"""

import numpy as np

class OfdmSym(object):
    
    """
    LTE DL OFDM Symbol Processing:class
    """

    def __init__(self, params):
        """
        LTE DL OFDM Symbol Processing:__init__
        - 3GPP TS 36.211 version 12.5.0 Release 12
            - 6.2 Slot structure and physical resource elements
                - Table 6.2.3-1: Physical resource blocks parameters
            - 6.12 OFDM baseband signal generation
                - Table 6.12-1: OFDM parameters
        """
        super(OfdmSym, self).__init__()
        # TODO: parametrization
        self.params = params
        # LTE parameters
        self.subcarriers = 12
        self.phy_res_blocks = [6, 15, 25, 50, 75, 100]
        self.fft_sizes = [128, 256, 512, 1024, 1536, 2048]
        self.bw_factors = [16, 8, 4, 2, 4.0/3.0, 1]
        # LTE DL derived parameters (Note! DC subcarrier!)
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
        self.rx_sym_start_indexes = np.array(self.cp + self.tx_sym_start_indexes)
        self.rx_sym_stop_indexes = np.array(self.rx_sym_start_indexes + self.n_fft)

    def tx(self, syms_fd):
        """
        LTE DL OFDM Symbol Processing:tx
        - Guard band and DC subcarrier insertion
        - IFFT processing
        - Cyclic prefix insertion
        - Optional: Windowing and/or filtering
        """
        # guard bands and zero "DC" subcarrier insertion
        syms_dc = np.zeros((self.n_ofdm_sym, 1), dtype=np.complex)
        syms_gb = np.zeros((self.n_ofdm_sym, self.n_gb), dtype=np.complex)
        syms_fd_shifted = np.hstack([syms_dc, syms_fd[:,self.n_subc/2:], syms_gb, syms_fd[:,:self.n_subc/2]])
        # IFFT processing with zero frequency term "DC" at a[0],
        syms_td = np.fft.ifft(syms_fd_shifted, n=self.n_fft, axis=1)
        # cyclic prefix insertion
        syms_cp = np.zeros(self.n_sym, dtype=np.complex)
        for sym_i in xrange(self.n_ofdm_sym):
            tmp = np.hstack(([syms_td[sym_i][-self.cp[sym_i]:], syms_td[sym_i]]))
            sym_start = self.tx_sym_start_indexes[sym_i]
            sym_stop = self.tx_sym_stop_indexes[sym_i]
            syms_cp[sym_start:sym_stop] = tmp
        return syms_cp

    def rx(self, syms_td, syms_ref):
        """
        LTE DL OFDM Symbol Processing:rx
        - Cyclic prefix removal
        - FFT processing
        - Guard band and DC subcarrier removal
        """
        # cyclic prefix removal
        syms_cp = np.zeros((self.n_ofdm_sym, self.n_fft), dtype=np.complex)
        for sym_i in xrange(self.n_ofdm_sym):
            sym_start = self.rx_sym_start_indexes[sym_i]
            sym_stop = self.rx_sym_stop_indexes[sym_i]
            syms_cp[sym_i,:] = syms_td[sym_start:sym_stop]
        # FFT processing
        syms_fd = np.fft.fft(syms_cp, axis=1)
        # guard band and DC subcarrier removal
        syms_fd_shifted = np.hstack([
            syms_fd[:,1+self.n_subc/2+self.n_gb:1+2*self.n_subc/2+self.n_gb],
            syms_fd[:,1:1+self.n_subc/2]])
        return syms_fd_shifted
