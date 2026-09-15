#!/usr/bin/env python

from ctypes import cdll, c_int
import numpy as np
import pandas as pd


class pyQoR:
    def __init__(self, qor_clib_path):
        self.qor_clib_path = qor_clib_path
        self.network_unstable = True
        self.wclatency = 10000000

    def setup_Clibs(self):
        self.lib = cdll.LoadLibrary(self.qor_clib_path)
        self.lib.main_west_fifo_network.argtypes = [c_int]
        self.lib.main_west_fifo_network_initialize.argtypes = [c_int]
        self.lib.main_west_fifo_network_terminate.argtypes = [c_int]
        self.lib.main_west_fifo_network_with_args.argtypes = [
            c_int,
            np.ctypeslib.ndpointer(dtype=np.double),
            np.ctypeslib.ndpointer(dtype=np.double),
            np.ctypeslib.ndpointer(dtype=np.double),
        ]
        self.lib.main_west_fifo_network_with_args.restype = c_int

    def reset_Clibs(self):
        self.lib.get_buffer.restype = np.ctypeslib.ndpointer(
            dtype=np.double, shape=(self.N * self.N,)
        )
        self.lib.get_wclatency.restype = np.ctypeslib.ndpointer(
            dtype=np.double, shape=(self.N * self.N * self.N * self.N)
        )
        self.lib.west_fifo_network_initialize()

    def setup_pe(self):
        self.pe_bm = np.zeros(
            (self.N * self.N, self.N * self.N), dtype=np.double
        )
        self.pe_rm = np.zeros(
            (self.N * self.N, self.N * self.N), dtype=np.double
        )
        self.length = len(self.df.index)

        for k in range(self.length):
            sx = int(self.df["sX"][k])
            sy = int(self.df["sY"][k])
            dx = int(self.df["dX"][k])
            dy = int(self.df["dY"][k])
            if self.burst_user is None:
                self.burst = np.double(self.df["B"][k])
            else:
                self.burst = np.double(self.burst_user)
            if self.rate_user is None:
                self.rate = np.double(self.df["R"][k])
            else:
                self.rate = self.rate_user

            self.pe_bm[sy * self.N + sx][dy * self.N + dx] = self.burst
            self.pe_rm[sy * self.N + sx][dy * self.N + dx] = self.rate

    def config_NoC(self, x):
        self.sw_bp = np.zeros(self.N * self.N, dtype=np.double)

        for i, el in enumerate(x):
            if el > 0:
                self.sw_bp[i] = 1
            else:
                self.sw_bp[i] = 0

    def calculate_route_stats(self):
        # TS_BL assumed to be 0, suffix hardcoded as 0
        self.stable = self.lib.main_west_fifo_network_with_args(
            self.N,
            np.ravel(self.pe_bm, order="F"),
            np.ravel(self.pe_rm, order="F"),
            self.sw_bp,
        )
        self.fread_burst = self.lib.get_buffer()
        self.fread_total = self.lib.get_wclatency()

        self.max_fifo_size = np.max(self.fread_burst)
        self.sum_fifo_size = np.sum(self.fread_burst)

        self.wclatency = np.max(self.fread_total)
        self.network_unstable = (
            self.stable == 0
            or np.where(np.isnan(self.fread_burst))[0].shape[0]
            or np.where(np.isnan(self.fread_total))[0].shape[0]
        )

        return self.wclatency, self.network_unstable

    def calculate_cost_stats(self):
        COST_FIFO_AND_SHADOW = 247
        COST_FIFO_ONLY = 161
        COST_SHADOW_ONLY = 189

        self.cost = 0

        if self.max_fifo_size <= 32:
            for i in range(self.N):
                cur_sum = np.sum(self.sw_bp[i * self.N : (i + 1) * self.N])
                if cur_sum == 0:
                    self.cost += COST_FIFO_ONLY * self.N
                else:
                    self.cost += (
                        cur_sum * COST_SHADOW_ONLY
                        + (self.N - cur_sum) * COST_FIFO_AND_SHADOW
                    )
        else:
            self.cost = 1000000000
            self.stable = 0

    def analyze_network(self, x, N, df, burst_user, rate_user):
        self.N = N
        self.df = df
        self.burst_user = burst_user
        self.rate_user = rate_user
        self.setup_Clibs()
        self.setup_pe()
        self.reset_Clibs()

        self.config_NoC(x)
        wclatency, network_unstable = self.calculate_route_stats()

        return wclatency if not network_unstable else 100000000
        # self.calculate_cost_stats()

    def close_lib(self):
        if self.operating_mode == "sharedlib":
            self.lib.west_fifo_network_terminate()        
