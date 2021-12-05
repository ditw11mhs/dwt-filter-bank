from re import T
from attr import s
import streamlit as st
import numpy as np
import scipy.signal as sps
import pandas as pd
import os
from icecream import ic


class Utils:
    def padding(self, x):
        log = np.log2(len(x))
        return np.pad(
            x, (0, int(2 ** ((log - log % 1) + 1) - len(x))), mode="constant"
        ).flatten()

    def FFT(self, x):

        if np.log2(len(x)) % 1 > 0:
            x = self.padding(x)

        x = np.asarray(x, dtype=float)
        N = x.shape[0]

        N_min = min(N, 2)

        # DFT on all length-N_min sub-problems at once
        n = np.arange(N_min)
        k = n[:, None]
        W = np.exp(-2j * np.pi * n * k / N_min)
        X = np.dot(W, x.reshape((N_min, -1)))

        # Recursive calculation all at once
        while X.shape[0] < N:
            X_even = X[:, : int(X.shape[1] / 2)]
            X_odd = X[:, int(X.shape[1] / 2) :]
            factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
            factor.shape, factor
            X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])
        return X.flatten()

    def DFT_FFT_magnitude_norm(self, X, fs):
        N = len(X)
        n = np.arange(N)
        f = (n * fs / N).flatten()
        X_norm = X
        X_norm[1:] = X_norm[1:] * 2
        X_norm = X_norm / N
        X_magnitude_norm = np.abs(X_norm)
        return {
            "Magnitude": np.array_split(X_magnitude_norm, 2)[0].flatten(),
            "Frequency": np.array_split(f, 2)[0].flatten(),
        }

    def FFT_mag(self, x, fs):
        X = self.FFT(x)
        return self.DFT_FFT_magnitude_norm(X, fs)

    @st.cache
    def one_data_reader(self):
        data_path = os.path.join("data", "ecg 100.dat")
        data_ecg = pd.read_csv(data_path, skiprows=[1], delimiter="\t")
        # split=12
        # return {
        #     "Time ECG": np.split(data_ecg["'Elapsed time'"].to_numpy().flatten(), split)[0],
        #     "V5": np.split(data_ecg["'V5'"].to_numpy().flatten(), split)[0],
        #     "MLII": np.split(data_ecg["'MLII'"].to_numpy().flatten(),split)[0],
        # }
        return {
            "Time ECG": data_ecg["'Elapsed time'"].to_numpy().flatten(),
            "V5": data_ecg["'V5'"].to_numpy().flatten(),
            "MLII": data_ecg["'MLII'"].to_numpy().flatten(),
        }

    @st.cache(allow_output_mutation=True)
    def two_data_reader(self):
        data_path = os.path.join("data", "flat3.TXT")
        data_gait = np.loadtxt(data_path)
        return {
            "Time": data_gait[:, 0],
            "Heel": data_gait[:, 1],
            "Toe": data_gait[:, 2],
            "Hip": data_gait[:, 3],
            "Knee": data_gait[:, 4],
            "Ankle": data_gait[:, 5],
        }

    def dwt_level(self, level):
        if level == 1:
            return np.array([-2, 2])
        elif level == 2:

            return np.array([-1, -3, -2, 2, 3, 1]) / 4
        elif level == 3:
            return np.array([-1, -3, -6, -10, -11, -9, -4, 4, 9, 11, 10, 6, 3, 1]) / 32
        elif level == 4:
            return (
                np.array(
                    [
                        -1,
                        -3,
                        -6,
                        -10,
                        -15,
                        -21,
                        -28,
                        -36,
                        -41,
                        -43,
                        -42,
                        -38,
                        -31,
                        -21,
                        -8,
                        8,
                        21,
                        31,
                        38,
                        42,
                        43,
                        41,
                        36,
                        28,
                        21,
                        15,
                        10,
                        6,
                        3,
                        1,
                    ]
                )
                / 256
            )
        elif level == 5:
            return (
                np.array(
                    [
                        -1,
                        -3,
                        -6,
                        -10,
                        -15,
                        -21,
                        -28,
                        -36,
                        -45,
                        -55,
                        -66,
                        -78,
                        -91,
                        -105,
                        -120,
                        -136,
                        -149,
                        -159,
                        -166,
                        -170,
                        -171,
                        -169,
                        -164,
                        -156,
                        -145,
                        -131,
                        -114,
                        -94,
                        -71,
                        -45,
                        -16,
                        16,
                        45,
                        71,
                        94,
                        114,
                        131,
                        145,
                        156,
                        164,
                        169,
                        171,
                        170,
                        166,
                        159,
                        149,
                        136,
                        120,
                        105,
                        91,
                        78,
                        66,
                        55,
                        45,
                        36,
                        28,
                        21,
                        15,
                        10,
                        6,
                        3,
                        1,
                    ]
                )
                / 2048
            )
        elif level == 6:
            return (
                np.array(
                    [
                        -1,
                        -3,
                        -6,
                        -10,
                        -15,
                        -21,
                        -28,
                        -36,
                        -45,
                        -55,
                        -66,
                        -78,
                        -91,
                        -105,
                        -120,
                        -136,
                        -153,
                        -171,
                        -190,
                        -210,
                        -231,
                        -253,
                        -276,
                        -300,
                        -325,
                        -351,
                        -378,
                        -406,
                        -435,
                        -465,
                        -496,
                        -528,
                        -557,
                        -583,
                        -606,
                        -626,
                        -643,
                        -657,
                        -668,
                        -676,
                        -681,
                        -683,
                        -682,
                        -678,
                        -671,
                        -661,
                        -648,
                        -632,
                        -613,
                        -591,
                        -566,
                        -538,
                        -507,
                        -473,
                        -436,
                        -396,
                        -353,
                        -307,
                        -258,
                        -206,
                        -151,
                        -93,
                        -32,
                        32,
                        93,
                        151,
                        206,
                        258,
                        307,
                        353,
                        396,
                        436,
                        473,
                        507,
                        538,
                        566,
                        591,
                        613,
                        632,
                        648,
                        661,
                        671,
                        678,
                        682,
                        683,
                        681,
                        676,
                        668,
                        657,
                        643,
                        626,
                        606,
                        583,
                        557,
                        528,
                        496,
                        465,
                        435,
                        406,
                        378,
                        351,
                        325,
                        300,
                        276,
                        253,
                        231,
                        210,
                        190,
                        171,
                        153,
                        136,
                        120,
                        105,
                        91,
                        78,
                        66,
                        55,
                        45,
                        36,
                        28,
                        21,
                        15,
                        10,
                        6,
                        3,
                        1,
                    ]
                )
                / 16384
            )
        elif level == 7:
            return (
                np.array(
                    [
                        -1,
                        -3,
                        -6,
                        -10,
                        -15,
                        -21,
                        -28,
                        -36,
                        -45,
                        -55,
                        -66,
                        -78,
                        -91,
                        -105,
                        -120,
                        -136,
                        -153,
                        -171,
                        -190,
                        -210,
                        -231,
                        -253,
                        -276,
                        -300,
                        -325,
                        -351,
                        -378,
                        -406,
                        -435,
                        -465,
                        -496,
                        -528,
                        -561,
                        -595,
                        -630,
                        -666,
                        -703,
                        -741,
                        -780,
                        -820,
                        -861,
                        -903,
                        -946,
                        -990,
                        -1035,
                        -1081,
                        -1128,
                        -1176,
                        -1225,
                        -1275,
                        -1326,
                        -1378,
                        -1431,
                        -1485,
                        -1540,
                        -1596,
                        -1653,
                        -1711,
                        -1770,
                        -1830,
                        -1891,
                        -1953,
                        -2016,
                        -2080,
                        -2141,
                        -2199,
                        -2254,
                        -2306,
                        -2355,
                        -2401,
                        -2444,
                        -2484,
                        -2521,
                        -2555,
                        -2586,
                        -2614,
                        -2639,
                        -2661,
                        -2680,
                        -2696,
                        -2709,
                        -2719,
                        -2726,
                        -2730,
                        -2731,
                        -2729,
                        -2724,
                        -2716,
                        -2705,
                        -2691,
                        -2674,
                        -2654,
                        -2631,
                        -2605,
                        -2576,
                        -2544,
                        -2509,
                        -2471,
                        -2430,
                        -2386,
                        -2339,
                        -2289,
                        -2236,
                        -2180,
                        -2121,
                        -2059,
                        -1994,
                        -1926,
                        -1855,
                        -1781,
                        -1704,
                        -1624,
                        -1541,
                        -1455,
                        -1366,
                        -1274,
                        -1179,
                        -1081,
                        -980,
                        -876,
                        -769,
                        -659,
                        -546,
                        -430,
                        -311,
                        -189,
                        -64,
                        64,
                        189,
                        311,
                        430,
                        546,
                        659,
                        769,
                        876,
                        980,
                        1081,
                        1179,
                        1274,
                        1366,
                        1455,
                        1541,
                        1624,
                        1704,
                        1781,
                        1855,
                        1926,
                        1994,
                        2059,
                        2121,
                        2180,
                        2236,
                        2289,
                        2339,
                        2386,
                        2430,
                        2471,
                        2509,
                        2544,
                        2576,
                        2605,
                        2631,
                        2654,
                        2674,
                        2691,
                        2705,
                        2716,
                        2724,
                        2729,
                        2731,
                        2730,
                        2726,
                        2719,
                        2709,
                        2696,
                        2680,
                        2661,
                        2639,
                        2614,
                        2586,
                        2555,
                        2521,
                        2484,
                        2444,
                        2401,
                        2355,
                        2306,
                        2254,
                        2199,
                        2141,
                        2080,
                        2016,
                        1953,
                        1891,
                        1830,
                        1770,
                        1711,
                        1653,
                        1596,
                        1540,
                        1485,
                        1431,
                        1378,
                        1326,
                        1275,
                        1225,
                        1176,
                        1128,
                        1081,
                        1035,
                        990,
                        946,
                        903,
                        861,
                        820,
                        780,
                        741,
                        703,
                        666,
                        630,
                        595,
                        561,
                        528,
                        496,
                        465,
                        435,
                        406,
                        378,
                        351,
                        325,
                        300,
                        276,
                        253,
                        231,
                        210,
                        190,
                        171,
                        153,
                        136,
                        120,
                        105,
                        91,
                        78,
                        66,
                        55,
                        45,
                        36,
                        28,
                        21,
                        15,
                        10,
                        6,
                        3,
                        1,
                    ]
                )
                / 131072
            )
        elif level == 8:
            return (
                np.array(
                    [
                        -1,
                        -3,
                        -6,
                        -10,
                        -15,
                        -21,
                        -28,
                        -36,
                        -45,
                        -55,
                        -66,
                        -78,
                        -91,
                        -105,
                        -120,
                        -136,
                        -153,
                        -171,
                        -190,
                        -210,
                        -231,
                        -253,
                        -276,
                        -300,
                        -325,
                        -351,
                        -378,
                        -406,
                        -435,
                        -465,
                        -496,
                        -528,
                        -561,
                        -595,
                        -630,
                        -666,
                        -703,
                        -741,
                        -780,
                        -820,
                        -861,
                        -903,
                        -946,
                        -990,
                        -1035,
                        -1081,
                        -1128,
                        -1176,
                        -1225,
                        -1275,
                        -1326,
                        -1378,
                        -1431,
                        -1485,
                        -1540,
                        -1596,
                        -1653,
                        -1711,
                        -1770,
                        -1830,
                        -1891,
                        -1953,
                        -2016,
                        -2080,
                        -2145,
                        -2211,
                        -2278,
                        -2346,
                        -2415,
                        -2485,
                        -2556,
                        -2628,
                        -2701,
                        -2775,
                        -2850,
                        -2926,
                        -3003,
                        -3081,
                        -3160,
                        -3240,
                        -3321,
                        -3403,
                        -3486,
                        -3570,
                        -3655,
                        -3741,
                        -3828,
                        -3916,
                        -4005,
                        -4095,
                        -4186,
                        -4278,
                        -4371,
                        -4465,
                        -4560,
                        -4656,
                        -4753,
                        -4851,
                        -4950,
                        -5050,
                        -5151,
                        -5253,
                        -5356,
                        -5460,
                        -5565,
                        -5671,
                        -5778,
                        -5886,
                        -5995,
                        -6105,
                        -6216,
                        -6328,
                        -6441,
                        -6555,
                        -6670,
                        -6786,
                        -6903,
                        -7021,
                        -7140,
                        -7260,
                        -7381,
                        -7503,
                        -7626,
                        -7750,
                        -7875,
                        -8001,
                        -8128,
                        -8256,
                        -8381,
                        -8503,
                        -8622,
                        -8738,
                        -8851,
                        -8961,
                        -9068,
                        -9172,
                        -9273,
                        -9371,
                        -9466,
                        -9558,
                        -9647,
                        -9733,
                        -9816,
                        -9896,
                        -9973,
                        -10047,
                        -10118,
                        -10186,
                        -10251,
                        -10313,
                        -10372,
                        -10428,
                        -10481,
                        -10531,
                        -10578,
                        -10622,
                        -10663,
                        -10701,
                        -10736,
                        -10768,
                        -10797,
                        -10823,
                        -10846,
                        -10866,
                        -10883,
                        -10897,
                        -10908,
                        -10916,
                        -10921,
                        -10923,
                        -10922,
                        -10918,
                        -10911,
                        -10901,
                        -10888,
                        -10872,
                        -10853,
                        -10831,
                        -10806,
                        -10778,
                        -10747,
                        -10713,
                        -10676,
                        -10636,
                        -10593,
                        -10547,
                        -10498,
                        -10446,
                        -10391,
                        -10333,
                        -10272,
                        -10208,
                        -10141,
                        -10071,
                        -9998,
                        -9922,
                        -9843,
                        -9761,
                        -9676,
                        -9588,
                        -9497,
                        -9403,
                        -9306,
                        -9206,
                        -9103,
                        -8997,
                        -8888,
                        -8776,
                        -8661,
                        -8543,
                        -8422,
                        -8298,
                        -8171,
                        -8041,
                        -7908,
                        -7772,
                        -7633,
                        -7491,
                        -7346,
                        -7198,
                        -7047,
                        -6893,
                        -6736,
                        -6576,
                        -6413,
                        -6247,
                        -6078,
                        -5906,
                        -5731,
                        -5553,
                        -5372,
                        -5188,
                        -5001,
                        -4811,
                        -4618,
                        -4422,
                        -4223,
                        -4021,
                        -3816,
                        -3608,
                        -3397,
                        -3183,
                        -2966,
                        -2746,
                        -2523,
                        -2297,
                        -2068,
                        -1836,
                        -1601,
                        -1363,
                        -1122,
                        -878,
                        -631,
                        -381,
                        -128,
                        128,
                        381,
                        631,
                        878,
                        1122,
                        1363,
                        1601,
                        1836,
                        2068,
                        2297,
                        2523,
                        2746,
                        2966,
                        3183,
                        3397,
                        3608,
                        3816,
                        4021,
                        4223,
                        4422,
                        4618,
                        4811,
                        5001,
                        5188,
                        5372,
                        5553,
                        5731,
                        5906,
                        6078,
                        6247,
                        6413,
                        6576,
                        6736,
                        6893,
                        7047,
                        7198,
                        7346,
                        7491,
                        7633,
                        7772,
                        7908,
                        8041,
                        8171,
                        8298,
                        8422,
                        8543,
                        8661,
                        8776,
                        8888,
                        8997,
                        9103,
                        9206,
                        9306,
                        9403,
                        9497,
                        9588,
                        9676,
                        9761,
                        9843,
                        9922,
                        9998,
                        10071,
                        10141,
                        10208,
                        10272,
                        10333,
                        10391,
                        10446,
                        10498,
                        10547,
                        10593,
                        10636,
                        10676,
                        10713,
                        10747,
                        10778,
                        10806,
                        10831,
                        10853,
                        10872,
                        10888,
                        10901,
                        10911,
                        10918,
                        10922,
                        10923,
                        10921,
                        10916,
                        10908,
                        10897,
                        10883,
                        10866,
                        10846,
                        10823,
                        10797,
                        10768,
                        10736,
                        10701,
                        10663,
                        10622,
                        10578,
                        10531,
                        10481,
                        10428,
                        10372,
                        10313,
                        10251,
                        10186,
                        10118,
                        10047,
                        9973,
                        9896,
                        9816,
                        9733,
                        9647,
                        9558,
                        9466,
                        9371,
                        9273,
                        9172,
                        9068,
                        8961,
                        8851,
                        8738,
                        8622,
                        8503,
                        8381,
                        8256,
                        8128,
                        8001,
                        7875,
                        7750,
                        7626,
                        7503,
                        7381,
                        7260,
                        7140,
                        7021,
                        6903,
                        6786,
                        6670,
                        6555,
                        6441,
                        6328,
                        6216,
                        6105,
                        5995,
                        5886,
                        5778,
                        5671,
                        5565,
                        5460,
                        5356,
                        5253,
                        5151,
                        5050,
                        4950,
                        4851,
                        4753,
                        4656,
                        4560,
                        4465,
                        4371,
                        4278,
                        4186,
                        4095,
                        4005,
                        3916,
                        3828,
                        3741,
                        3655,
                        3570,
                        3486,
                        3403,
                        3321,
                        3240,
                        3160,
                        3081,
                        3003,
                        2926,
                        2850,
                        2775,
                        2701,
                        2628,
                        2556,
                        2485,
                        2415,
                        2346,
                        2278,
                        2211,
                        2145,
                        2080,
                        2016,
                        1953,
                        1891,
                        1830,
                        1770,
                        1711,
                        1653,
                        1596,
                        1540,
                        1485,
                        1431,
                        1378,
                        1326,
                        1275,
                        1225,
                        1176,
                        1128,
                        1081,
                        1035,
                        990,
                        946,
                        903,
                        861,
                        820,
                        780,
                        741,
                        703,
                        666,
                        630,
                        595,
                        561,
                        528,
                        496,
                        465,
                        435,
                        406,
                        378,
                        351,
                        325,
                        300,
                        276,
                        253,
                        231,
                        210,
                        190,
                        171,
                        153,
                        136,
                        120,
                        105,
                        91,
                        78,
                        66,
                        55,
                        45,
                        36,
                        28,
                        21,
                        15,
                        10,
                        6,
                        3,
                        1,
                    ]
                )
                / 1048576
            )

    def conv(self, x, level):
        data_conv = sps.fftconvolve(x, self.dwt_level(level), mode="same")
        # data_conv = data_conv[::2*level]
        return data_conv

    def split(self, x, split):
        return x[split[0] : split[1]]

    def splitter(self, split, dic):
        dic["C1"] = self.split(dic["C1"], split)
        dic["C2"] = self.split(dic["C2"], split)
        dic["C3"] = self.split(dic["C3"], split)
        dic["C4"] = self.split(dic["C4"], split)
        dic["C5"] = self.split(dic["C5"], split)
        dic["C6"] = self.split(dic["C6"], split)
        dic["C7"] = self.split(dic["C7"], split)
        dic["C8"] = self.split(dic["C8"], split)
        dic["Time ECG"] = self.split(dic["Time ECG"], split)
        dic["Data"] = self.split(dic["Data"], split)
        return dic

    def shift(self, x, amount, direction):
        if amount > 0:
            out = np.pad(x, (0, amount), "constant")[amount:]
        elif amount < 0:
            out = np.pad(x, (abs(amount), 0), "constant")[:amount]
        elif amount == 0:
            return x
        return out

    def thresholding(self, x, threshold):
        out = (np.abs(x) > threshold) * 1
        out = self.moving_average(out, 5)
        out = (out > (0.5)) * 1
        return out

    def thresholding_t4(self, x, threshold):
        out = np.abs(x)
        out = self.moving_average(out, 2)
        out = (out > threshold) * 1
        # out = np.sign(x)
        # out = np.diff(out)
        # out = (out>0)*1
        # ic(np.where(out)[0])
        # ic(sps.find_peaks(x,height=threshold)[0].shape)
        # ic(out[:,None].shape)
        # ic(sps.find_peaks(x,height=threshold)[0].reshape(1,-1).shape)
        # ic(np.sum((np.where(out)[0].reshape(-1,1)>sps.find_peaks(x,height=threshold)[0].reshape(1,-1))*1))
        
        # ic(sps.find_peaks(x,height=threshold)[0])
        # ic(np.roll(sps.find_peaks(x,height=threshold)[0],-1))
        # ic(np.argwhere(np.logical_and(np.where(out)[0][:,None]>=sps.find_peaks(np.abs(x),height=threshold)[0],np.where(out)[0][:,None]<=np.roll(sps.find_peaks(np.abs(x),height=threshold)[0],-1))*1==1))
        
        
        return out

    

    def thresholding_t5(self, x, threshold):
        out = np.abs(x)
        out = self.moving_average(out, 2)
        out = (out > threshold) * 1
        # out = self.moving_average(out, 5)
        # out = (out > (1.25)) * 1
        return out

    def and_block(self, input_list, and_number):
        out = input_list[0]
        for i in range(1, and_number):
            out = np.logical_and(out, input_list[i])
        out = out * 1
        return out

    def moving_average(self, a, n=3):
        return sps.fftconvolve(a, np.ones(n), "same") / 3


class Solver(Utils):
    def __init__(self):
        self.one_data = self.one_data_reader()
        self.two_data = self.two_data_reader()

    def parallel_dwt(self, data_name):
        x = self.one_data[data_name]
        parallel_dwt_dict = {
            "C1": self.conv(x, 1),
            "C2": self.conv(x, 2),
            "C3": self.conv(x, 3),
            "C4": self.conv(x, 4),
            "C5": self.conv(x, 5),
            "C6": self.conv(x, 6),
            "C7": self.conv(x, 7),
            "C8": self.conv(x, 8),
            "Time ECG": self.one_data["Time ECG"],
            "Data": self.one_data[data_name],
            "Data Name": data_name,
        }
        return parallel_dwt_dict

    def filter_block(self, oc, inp, T, N):
        inp = np.pad(inp, (3, 0), "constant")
        out = np.zeros(N + 3)

        for n in range(N):
            out[n + 3] = (
                oc ** 3 * inp[n + 3]
                + 3 * oc ** 3 * inp[n + 2]
                + 3 * oc ** 3 * inp[n + 1]
                + oc ** 3 * inp[n]
                - ((-24 / T ** 3) - (8 / T ** 2*oc) + (4 / T * oc ** 2) + 3 * oc ** 3)
                * out[n + 2]
                - ((24 / T ** 3) - (8 / T ** 2*oc) - (4 / T * oc ** 2) + 3 * oc ** 3)
                * out[n + 1]
                - ((-8 / T ** 3) + (8 / T ** 2*oc) - (4 / T * oc ** 2) + oc ** 3) * out[n]
            )
            out[n + 3] = out[n + 3] / (
                (8 / T ** 3) + (8 / T ** 2*oc) + (4 / T * oc ** 2) + oc ** 3
            )
        return out[3:]

    def filter_forward_backward(self, data_input, fc):
        omega_c = 2 * np.pi * fc
        N_input = len(data_input)
        fs = 1000
        T = 1 / fs
        
        forward = self.filter_block(omega_c, data_input, T, N_input)
        backward = self.filter_block(omega_c, np.flip(forward), T, N_input)
        return {
            "Forward": forward,
            "Backward": np.flip(backward),
        }

    @st.cache
    def one_solver_a(self, data_name, split):
        dictionary = self.parallel_dwt(data_name)
        out = self.splitter(split, dictionary)
        return out

    def one_solver_b_searcher(self, search_input):
        starting_dict = self.parallel_dwt(search_input["Data Name"])
        T_select = search_input["T Selection"][1]
        delay = search_input["T" + str(T_select)]
        threshold = search_input["Threshold " + str(T_select)]
        coef = starting_dict["C" + str(T_select)]
        # Apply Delay
        starting_dict["C" + str(T_select)] = self.shift(coef, int(delay), "left")
        # Apply Splitter
        out_dict = self.splitter(search_input["Search Split"], starting_dict)
        # Get Coefficient Threshold
        if T_select == "4":
            print("asdasdasd")
            out_dict["Threshold Out"] = self.thresholding_t4(
                starting_dict["C" + str(T_select)], threshold
            )
        elif T_select == "5":
            out_dict["Threshold Out"] = self.thresholding_t5(
                starting_dict["C" + str(T_select)], threshold
            )
        else:
            out_dict["Threshold Out"] = self.thresholding(
                starting_dict["C" + str(T_select)], threshold
            )
        out_dict["T Selection"] = search_input["T Selection"]
        return out_dict

    def one_solver_b_segmentation(self, segmentation_input):
        starting_dict = self.parallel_dwt(segmentation_input["Data Name"])
        split = segmentation_input["Split Segment"]

        for i in range(1, 6):
            if i == 1:
                delay = segmentation_input["T5"] - segmentation_input["T1"]
            elif i == 2:
                delay = segmentation_input["T5"] - segmentation_input["T2"]
            elif i == 3:
                delay = segmentation_input["T5"] - segmentation_input["T3"]
            elif i == 4:
                delay = segmentation_input["T5"] - segmentation_input["T4"]
            elif i == 5:
                delay = segmentation_input["T5"]

            threshold = segmentation_input["Threshold " + str(i)]
            coef = starting_dict["C" + str(i)]

            if i == 4:
                starting_dict["C" + str(i) + " Processed"] = self.thresholding_t4(
                    coef, threshold
                )
            elif i == 5:
                starting_dict["C" + str(i) + " Processed"] = self.thresholding_t5(
                    coef, threshold
                )
            else:
                starting_dict["C" + str(i) + " Processed"] = self.thresholding(
                    coef, threshold
                )

            starting_dict["C" + str(i) + " Processed"] = self.shift(
                starting_dict["C" + str(i) + " Processed"], int(delay), "left"
            )

        c1_p = starting_dict["C1 Processed"]
        c2_p = starting_dict["C2 Processed"]
        c3_p = starting_dict["C3 Processed"]
        c4_p = starting_dict["C4 Processed"]
        c5_p = starting_dict["C5 Processed"]

        # QRS Logic
        qrs_input = [c1_p, c2_p, c3_p, c4_p, c5_p]
        starting_dict["QRS"] = self.and_block(qrs_input, 5)

        # T Wave Logic
        not_qrs = np.logical_not(starting_dict["QRS"])
        t_input = [not_qrs, c4_p, c5_p]
        starting_dict["T Wave"] = self.and_block(t_input, 3)

        # P Wave Logic
        not_t = np.logical_not(starting_dict["T Wave"])
        p_input = [not_t, not_qrs, c4_p]
        starting_dict["P Wave"] = self.and_block(p_input, 3)

        starting_dict["PQRST"] = (
            starting_dict["QRS"] + starting_dict["T Wave"] + starting_dict["P Wave"]
        )

        starting_dict["QRS"] = self.split(starting_dict["QRS"], split)
        starting_dict["T Wave"] = self.split(starting_dict["T Wave"], split)
        starting_dict["P Wave"] = self.split(starting_dict["P Wave"], split)
        starting_dict["PQRST"] = self.split(starting_dict["PQRST"], split)
        starting_dict = self.splitter(split, starting_dict)

        return starting_dict

    def two_solver(self, data_input):
        data = self.two_data
        out = data
        out["Heel FFT"] = self.FFT_mag(data["Heel"], 1000)
        out["Toe FFT"] = self.FFT_mag(data["Toe"], 1000)
        out["Hip FFT"] = self.FFT_mag(data["Hip"], 1000)
        out["Knee FFT"] = self.FFT_mag(data["Knee"], 1000)
        out["Ankle FFT"] = self.FFT_mag(data["Ankle"], 1000)
        
        out["Filtered Heel"] = self.filter_forward_backward(out["Heel"],data_input["fc Heel"])
        out["Filtered Toe"] = self.filter_forward_backward(out["Toe"],data_input["fc Toe"])
        out["Filtered Hip"] = self.filter_forward_backward(out["Hip"],data_input["fc Hip"])
        out["Filtered Knee"] = self.filter_forward_backward(out["Knee"],data_input["fc Knee"])
        out["Filtered Ankle"] = self.filter_forward_backward(out["Ankle"],data_input["fc Ankle"])
        
        return out
