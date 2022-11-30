# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:38:09 2021

@author: noppen
"""

import numpy as np
import rioxarray as rio
import os
from scipy.ndimage import convolve
from rasterio.fill import fillnodata
from skimage.morphology import skeletonize
from scipy.ndimage.morphology import binary_hit_or_miss
from scipy.ndimage import convolve
from scipy.ndimage import rotate
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import time
from tqdm.auto import tqdm
from pathlib import Path


class DetectLineElements:
    """
    Class containing three algorithms to detect high line elements in a DTM for 2D-DHYDRO modelling.
    """

    def __init__(self, path):
        self.root = Path(path)

    def algorithm_1(
        self,
        raster_in=None,
        result_path=None,
        N_theta=None,
        variable_sigma=None,
        threshold=None,
    ):
        """
        Algorithm 1 of Thirza van Noppen's thesis to detect high line elements in a DTM.

        Parameters
        ----------
        raster_in : str
            AHN raster file name.
        result_path : WindowsPath, optional
            Path to write the output. Default is the root path.
        N_theta : int, optional
            Number of orientations. Default is 6.
        variable_sigma : array, optional
            Kernel size. The default is array([0.2, 0.4, 0.6, 0.8, 1. ])
        threshold : int, optional
            Threshold value. The default is 15

        Returns
        -------
        None. Four TIF-files are written containing the output.

        """
        if raster_in is None:
            raster_in = self.root / "AHN" / str("AHN_TEST.TIF")
        if result_path is None:
            print("No result folder provided, writing it to root.")
            result_path = self.root

        # Parameters voor kernels
        if N_theta is None:
            N_theta = 6  # Number of orientations
        variable_theta = np.arange(0, np.pi, np.pi / N_theta)
        if variable_sigma is None:
            variable_sigma = np.arange(0.2, 1.1, 0.2)  # Size kernel
        if threshold is None:
            threshold = 15  # Threshold value

        # Lookup (l_nms) for non-maximum suppression
        # Here the l_nms indicates the number of pixels that are considered when drawing a line perpendicular to the ridge structure
        l_nms = 4

        # Lookup (l_c) for connection points and line ends
        l_c = 3

        #%% PART I: PRE-PROCESSING DATA
        # Import DTM
        z = rio.open_rasterio(raster_in)

        # Select data AHN
        tif = z[0, :, :]
        sample = np.where(tif > 1e5, np.nan, tif.values)

        # IDW > Fill nan values
        s_mask = np.where(np.isnan(sample), 0, 1)
        sample = fillnodata(sample, mask=s_mask)

        #% PART II: KERNEL CONVOLUTION
        def kernel_conv(DEM, s=1, t=0, r=1, lowest=-5, highest=5, step=0.1, gamma=0.75):
            x = np.arange(highest, lowest, -step)
            y = np.arange(lowest, highest, step)
            x, y = np.meshgrid(x, y)

            # Create kernel based on sigma, theta & rho
            M_theta = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            rho_sqrt = np.array([[r ** 2, 0], [0, r ** -2]])
            z_second = np.zeros((len(x), len(y)))

            for i in range(len(x)):
                for j in range(len(y)):
                    xy = np.array([x[i, j], y[i, j]])
                    x_y = np.array([[x[i, j]], [y[i, j]]])
                    phi = xy @ M_theta.transpose() @ rho_sqrt @ M_theta @ x_y
                    G_hat = (1 / (2 * np.pi * s ** 2)) * np.exp(-phi / (2 * s ** 2))
                    SAG = (
                        (((x[i, j] * np.cos(t)) + (y[i, j] * np.sin(t))) ** 2)
                        / (r ** -4 * s ** 4)
                        - (r ** 2 / s ** 2)
                    ) * G_hat
                    z_second[i, j] = SAG

            # Convolution
            DEM_conv = -((s ** 2) ** gamma) * convolve(DEM, z_second)
            return DEM_conv

        # Sigma van 1: resulteert bij step=0.1 in een positieve piek van 20 stappen, omgerekend naar AHN(0.5m) geeft detectie ridge van: 10 meter
        # Keuze voor detecteren ridge van 3 tot 15 meter breed geeft sigma: 0.3 tot 1.5
        Direction = np.zeros(np.shape(sample))
        Sigma_final = np.zeros(np.shape(sample)) + variable_sigma[0]

        for j in range(len(variable_sigma)):
            print(j)
            for k in range(len(variable_theta)):
                DTM_conv = kernel_conv(
                    DEM=sample, s=variable_sigma[j], t=variable_theta[k]
                )
                if k == 0 and j == 0:
                    Result_conv = DTM_conv  # Wordt uiteindelijke resultaat in opgeslagen van convolutie
                else:
                    Sigma_final = np.where(
                        DTM_conv > Result_conv, variable_sigma[j], Sigma_final
                    )
                    Direction = np.where(
                        DTM_conv > Result_conv, variable_theta[k], Direction
                    )
                    Result_conv = np.where(
                        DTM_conv > Result_conv, DTM_conv, Result_conv
                    )

        # PART III: THRESHOLDING
        Result_binary = np.where(Result_conv > threshold, 1, 0)

        # PART IV: NON-MAXIMA SUPPRESSION
        x, y = np.shape(sample)
        DTM_extra = np.empty(
            [(2 * l_nms + sample.shape[0]), (2 * l_nms + sample.shape[1])]
        )
        DTM_extra[:] = np.nan
        DTM_extra[l_nms:-l_nms, l_nms:-l_nms] = Result_conv

        NMS = np.empty(np.shape(DTM_extra))
        NMS[:] = np.nan
        NMS[l_nms:-l_nms, l_nms:-l_nms] = Result_conv

        Line = np.zeros(np.shape(DTM_extra))

        for i in range(x):
            print(i)
            for j in range(y):
                if Result_binary[i, j] == 1:
                    theta = Direction[i, j]
                    a_1 = np.round(l_nms * np.cos(theta)) + l_nms
                    a_2 = np.round(l_nms * -np.cos(theta)) + l_nms
                    b_1 = np.round(l_nms * -np.sin(theta)) + l_nms
                    b_2 = np.round(l_nms * np.sin(theta)) + l_nms

                    im = np.asarray(Line, dtype=np.uint8)
                    im = Image.fromarray(im, mode="L")
                    draw = ImageDraw.Draw(im)
                    draw.line(((a_1 + j), (b_1 + i), (a_2 + j), (b_2 + i)), fill=1)
                    c = np.array(im)
                    arr = DTM_extra[np.where(c == 1)]

                    if DTM_extra[i + l_nms, j + l_nms] < (np.nanmax(arr)):
                        NMS[i + l_nms, j + l_nms] = np.nan

        Result_NMS = NMS[l_nms:-l_nms, l_nms:-l_nms]
        Result_NMS = np.where(Result_binary == 0, np.nan, Result_NMS)
        Result_binary = np.where(Result_NMS > 0, 1, 0)

        # Outside to np.nan
        Result_binary[0, :] = 0
        Result_binary[-1, :] = 0
        Result_binary[:, 0] = 0
        Result_binary[:, -1] = 0

        # PART V: CONNECTION POINTS TO LINES
        a = 2
        b = 3
        ridge_extra = np.zeros([(2 * l_c + x), (2 * l_c + y)])
        endpoints = np.zeros([(2 * l_c + x), (2 * l_c + y)])
        ridge_extra[l_c:-l_c, l_c:-l_c] = Result_binary

        # Function to connect lines from point i,j to point x, y)
        # Input is neighbors (NB), coordinates point 1 (i,j), Binary result (RE) and distance wherein lines/points are connected
        def connection(NB, i, j, RE, n):
            a = np.zeros(np.shape(RE))
            [x_coor, y_coor] = np.where(NB > 0)
            x_coor = x_coor + i - n
            y_coor = y_coor + j - n

            for k in range(len(x_coor)):
                a = np.asarray(a, dtype=np.uint8)
                im = Image.fromarray(a, mode="L")
                draw = ImageDraw.Draw(im)

                draw.line((j, i, y_coor[k], x_coor[k]), fill=1)
                np.array(im)
                RE = im + RE
                RE = np.where(RE > 0, 1, 0)  # Binary

                a = np.zeros(np.shape(RE))
            return RE

        # Create 4 options of 3x3 kernel that can be used to find the end lines
        arr1 = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
        arr2 = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
        arr3 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        arr4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

        # Find all line ends by hit or miss with created 3x3 kernel
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr1) + endpoints
            arr1 = rotate(arr1, angle=90)
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr2) + endpoints
            arr2 = rotate(arr2, angle=90)
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr3) + endpoints
            arr3 = rotate(arr3, angle=90)
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr4) + endpoints
            arr4 = rotate(arr4, angle=90)

        # Connect all line ends and separate point that are within a distince of length l
        # values a and b are used to create local neighbors
        for i in range(l_c, x + l_c):
            for j in range(l_c, y + l_c):
                if ridge_extra[i, j] == 1:

                    # Neighboring points
                    neighbors = ridge_extra[i - 1 : i + 2, j - 1 : j + 2]
                    if np.sum(neighbors) == 1:
                        a = 2
                        b = 3

                        if (
                            np.sum(
                                ridge_extra[
                                    (i - l_c) : (i + (l_c + 1)),
                                    (j - l_c) : (j + (l_c + 1)),
                                ]
                            )
                            == 1
                        ):
                            ridge_extra[i, j] = 0

                        for k in range(l_c):
                            if (
                                np.sum(
                                    ridge_extra[(i - a) : (i + b), (j - a) : (j + b)]
                                )
                                > 1
                            ):
                                neighbors = ridge_extra[
                                    (i - a) : (i + b), (j - a) : (j + b)
                                ]
                                ridge_extra = connection(
                                    neighbors, i, j, ridge_extra, n=a
                                )
                            a = a + 1
                            b = b + 1

                    # Connection line ends
                    if endpoints[i, j] == 1:
                        a = 2
                        b = 3
                        for k in range(l_c):
                            if (
                                np.sum(endpoints[(i - a) : (i + b), (j - a) : (j + b)])
                                == 2
                            ):
                                neighbors = endpoints[
                                    (i - a) : (i + b), (j - a) : (j + b)
                                ]
                                endpoints = connection(neighbors, i, j, endpoints, n=a)
                            a = a + 1
                            b = b + 1

        ridge_extra = ridge_extra + endpoints
        Result_binary = np.where(
            ridge_extra[l_c:-l_c, l_c:-l_c] > 0, 1, np.nan
        )  # Binary
        Result = np.where(
            Result_binary == 1, Result_conv, np.nan
        )  # Fill with values convolution

        end = time.time()
        print(end - start)

        # Thinning lines for further processing in qgis
        Result_binary = skeletonize(Result_binary)
        Result_binary = np.where(Result_binary == 1, 1, np.nan)

        # """SAVE RESULTS"""
        tif = z
        tif[0, :, :] = Result_binary
        tif.rio.to_raster(result_path / str("A1_Results_binary.tif"))

        tif = z
        tif[0, :, :] = Result
        tif.rio.to_raster(result_path / str("A1_Results.tif"))

        tif = z
        tif[0, :, :] = Sigma_final
        tif.rio.to_raster(result_path / str("A1_sigma.tif"))

        tif = z
        tif[0, :, :] = Result_conv
        tif.rio.to_raster(result_path / str("A1_results_convolution.tif"))

    #%% ----------------------------------------------------------------------
    def algorithm_2(
        self,
        raster_in=None,
        result_path=None,
        N_theta=None,
        variable_sigma=None,
        variable_rho=None,
        threshold=None,
    ):
        """
        Algorithm 2 of Thirza van Noppen's thesis to detect high line elements in a DTM.

        Parameters
        ----------
        raster_in : str
            AHN raster file name.
        result_path : WindowsPath, optional
            Path to write the output. Default is the root path.
        N_theta : int, optional
            Number of orientations. Default is 6.
        variable_sigma : array, optional
            Kernel size. The default is array([0.2, 0.4, 0.6, 0.8, 1. ])
        threshold : int, optional
            Threshold value. The default is 15

        Returns
        -------
        None. Four TIF-files are written containing the output.

        """
        if raster_in is None:
            raster_in = self.root / "AHN" / str("AHN_TEST.TIF")
        if result_path is None:
            print("No result folder provided, writing it to root.")
            result_path = self.root

        # Parameters voor kernels
        if N_theta is None:
            N_theta = 6  # Number of orientations
        variable_theta = np.arange(0, np.pi, np.pi / N_theta)
        if variable_sigma is None:
            variable_sigma = np.arange(0.2, 1.1, 0.2)  # Size kernel
        if variable_rho is None:
            variable_rho = np.arange(1, 1.6, 0.2)
        if threshold is None:
            threshold = 15  # Threshold value
        start = time.time()

        threshold = 30  # Threshold value

        # Lookup (l_nms) for non-maximum suppression
        # Here the l_nms indicates the number of pixels that are considered when drawing a line perpendicular to the ridge structure
        l_nms = 4

        # Lookup (l_c) for connection points and line ends
        l_c = 3

        # PART I: PRE-PROCESSING DATA
        # Import DTM
        z = rio.open_rasterio(raster_in)

        # Select data AHN
        tif = z[0, :, :]
        sample = np.where(tif > 1e5, np.nan, tif.values)

        # Inverse Distance Weighting
        s_mask = np.where(np.isnan(sample), 0, 1)
        sample = fillnodata(sample, mask=s_mask)

        # PART II: KERNEL CONVOLUTION
        def kernel_conv(DEM, s=1, t=0, r=1, lowest=-6, highest=6, step=0.1, gamma=0.75):
            x = np.arange(highest, lowest, -step)
            y = np.arange(lowest, highest, step)
            x, y = np.meshgrid(x, y)

            # Create kernel based on sigma, theta & rho
            M_theta = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            rho_sqrt = np.array([[r ** 2, 0], [0, r ** -2]])
            second = np.zeros((len(x), len(y)))
            for i in range(len(x)):
                for j in range(len(y)):
                    xy = np.array([x[i, j], y[i, j]])
                    x_y = np.array([[x[i, j]], [y[i, j]]])
                    phi = xy @ M_theta.transpose() @ rho_sqrt @ M_theta @ x_y
                    G_hat = (1 / (2 * np.pi * s ** 2)) * np.exp(-phi / (2 * s ** 2))
                    SAG = (
                        (((x[i, j] * np.cos(t)) + (y[i, j] * np.sin(t))) ** 2)
                        / (r ** -4 * s ** 4)
                        - (r ** 2 / s ** 2)
                    ) * G_hat
                    second[i, j] = SAG

            # Convolution
            DEM_conv = -(s ** (2 * gamma)) * convolve(DEM, second)
            return DEM_conv

        def kernel_conv2(
            DEM, s=1, t=0, r=1, lowest=-6, highest=6, step=0.1, gamma=0.75
        ):
            x = np.arange(highest, lowest, -step)
            y = np.arange(lowest, highest, step)
            x, y = np.meshgrid(x, y)

            # Create kernel based on sigma, theta & rho
            M_theta = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            rho_sqrt = np.array([[r ** 2, 0], [0, r ** -2]])
            rhotheta = np.matmul(rho_sqrt, M_theta)
            second = np.zeros((len(x), len(y)))
            for j in range(len(y)):
                xy2 = np.array([x[:, j], y[:, j]])
                x_y = np.array([[x[i, j]], [y[:, j]]])
                phi2 = np.diagonal(
                    np.dot(np.dot(xy2.T, M_theta.transpose()), np.dot(rhotheta, xy2))
                )
                G_hat2 = (1 / (2 * np.pi * s ** 2)) * np.exp(-phi2 / (2 * s ** 2))
                SAG2 = (
                    (((x[:, j] * np.cos(t)) + (y[:, j] * np.sin(t))) ** 2)
                    / (r ** -4 * s ** 4)
                    - (r ** 2 / s ** 2)
                ) * G_hat2
                second[:, j] = SAG2

            # Convolution
            DEM_conv = -(s ** (2 * gamma)) * convolve(DEM, second)
            return DEM_conv

        def computeCONV(i, j, k):
            DTM_conv = kernel_conv2(
                DEM=sample, s=variable_sigma[j], t=variable_theta[k], r=variable_rho[i]
            )

            return DTM_conv

        def procesCONVS(
            DTM_convs, Direction, Sigma_final, Rho_final, Result_conv, i, j
        ):
            for k, DTM_conv in enumerate(DTM_convs):
                if j == 0 and k == 0:
                    Result_conv = DTM_conv
                Direction = np.where(
                    DTM_conv > Result_conv, variable_theta[k], Direction
                )
                Sigma_final = np.where(
                    DTM_conv > Result_conv, variable_sigma[j], Sigma_final
                )
                Rho_final = np.where(DTM_conv > Result_conv, variable_rho[i], Rho_final)
                Result_conv = np.where(DTM_conv > Result_conv, DTM_conv, Result_conv)
            return Direction, Sigma_final, Rho_final, Result_conv

        Direction = np.zeros(np.shape(sample))
        Sigma_final = np.zeros(np.shape(sample)) + variable_sigma[0]
        Rho_final = np.ones(np.shape(sample)) + variable_rho[0]
        Result_conv = None

        t = time.time()
        for i in tqdm(range(len(variable_rho)), total=len(variable_rho)):
            for j in range(len(variable_sigma)):
                DTM_convs = Parallel(n_jobs=8)(
                    delayed(computeCONV)(i, j, k) for k in range(len(variable_theta))
                )
                Direction, Sigma_final, Rho_final, Result_conv = procesCONVS(
                    DTM_convs, Direction, Sigma_final, Rho_final, Result_conv, i, j
                )

        elapsed = time.time() - t
        print(elapsed)

        #%% PART III: THRESHOLDING
        Result_binary = np.where(Result_conv > threshold, 1, 0)

        # PART IV: NON-MAXIMA SUPPRESSION
        x, y = np.shape(sample)
        DTM_extra = np.empty(
            [(2 * l_nms + sample.shape[0]), (2 * l_nms + sample.shape[1])]
        )
        DTM_extra[:] = np.nan
        DTM_extra[l_nms:-l_nms, l_nms:-l_nms] = Result_conv

        NMS = np.empty(np.shape(DTM_extra))
        NMS[:] = np.nan
        NMS[l_nms:-l_nms, l_nms:-l_nms] = Result_conv

        Line = np.zeros(np.shape(DTM_extra))

        for i in range(x):
            print(i)
            for j in range(y):
                if Result_binary[i, j] == 1:
                    theta = Direction[i, j]
                    a_1 = np.round(l_nms * np.cos(theta)) + l_nms
                    a_2 = np.round(l_nms * -np.cos(theta)) + l_nms
                    b_1 = np.round(l_nms * -np.sin(theta)) + l_nms
                    b_2 = np.round(l_nms * np.sin(theta)) + l_nms

                    im = np.asarray(Line, dtype=np.uint8)
                    im = Image.fromarray(im, mode="L")
                    draw = ImageDraw.Draw(im)
                    draw.line(((a_1 + j), (b_1 + i), (a_2 + j), (b_2 + i)), fill=1)
                    c = np.array(im)
                    arr = DTM_extra[np.where(c == 1)]

                    if DTM_extra[i + l_nms, j + l_nms] < (np.nanmax(arr)):
                        NMS[i + l_nms, j + l_nms] = np.nan

        Result_NMS = NMS[l_nms:-l_nms, l_nms:-l_nms]
        Result_NMS = np.where(Result_binary == 0, np.nan, Result_NMS)
        Result_binary = np.where(Result_NMS > 0, 1, 0)

        # PART V: CONNECTION POINTS TO LINES
        a = 2
        b = 3
        ridge_extra = np.zeros([(2 * l_c + x), (2 * l_c + y)])
        endpoints = np.zeros([(2 * l_c + x), (2 * l_c + y)])
        ridge_extra[l_c:-l_c, l_c:-l_c] = Result_binary

        # Function to connect lines from point i,j to point x, y)
        # Input is neighbors (NB), coordinates point 1 (i,j), Binary result (RE) and distance wherein lines/points are connected
        def connection(NB, i, j, RE, n):
            a = np.zeros(np.shape(RE))
            [x_coor, y_coor] = np.where(NB > 0)
            x_coor = x_coor + i - n
            y_coor = y_coor + j - n

            for k in range(len(x_coor)):
                a = np.asarray(a, dtype=np.uint8)
                im = Image.fromarray(a, mode="L")
                draw = ImageDraw.Draw(im)

                draw.line((j, i, y_coor[k], x_coor[k]), fill=1)
                np.array(im)
                RE = im + RE
                RE = np.where(RE > 0, 1, 0)  # Binary

                a = np.zeros(np.shape(RE))
            return RE

        # Create 4 options of 3x3 kernel that can be used to find the end lines
        arr1 = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
        arr2 = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
        arr3 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        arr4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

        # Find all line ends by hit or miss with created 3x3 kernel
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr1) + endpoints
            arr1 = rotate(arr1, angle=90)
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr2) + endpoints
            arr2 = rotate(arr2, angle=90)
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr3) + endpoints
            arr3 = rotate(arr3, angle=90)
        for i in range(4):
            endpoints = binary_hit_or_miss(ridge_extra, arr4) + endpoints
            arr4 = rotate(arr4, angle=90)

        # Connect all line ends and separate point that are within a distince of length l
        # values a and b are used to create local neighbors

        for i in range(l_c, x + l_c):
            for j in range(l_c, y + l_c):
                if ridge_extra[i, j] == 1:

                    # Neighboring points
                    neighbors = ridge_extra[i - 1 : i + 2, j - 1 : j + 2]
                    if np.sum(neighbors) == 1:
                        a = 2
                        b = 3

                        if (
                            np.sum(
                                ridge_extra[
                                    (i - l_c) : (i + (l_c + 1)),
                                    (j - l_c) : (j + (l_c + 1)),
                                ]
                            )
                            == 1
                        ):
                            ridge_extra[i, j] = 0

                        for k in range(l_c):
                            if (
                                np.sum(
                                    ridge_extra[(i - a) : (i + b), (j - a) : (j + b)]
                                )
                                > 1
                            ):
                                neighbors = ridge_extra[
                                    (i - a) : (i + b), (j - a) : (j + b)
                                ]
                                ridge_extra = connection(
                                    neighbors, i, j, ridge_extra, n=a
                                )
                            a = a + 1
                            b = b + 1

                    # Connection line ends
                    if endpoints[i, j] == 1:
                        a = 2
                        b = 3
                        for k in range(l_c):
                            if (
                                np.sum(endpoints[(i - a) : (i + b), (j - a) : (j + b)])
                                == 2
                            ):
                                neighbors = endpoints[
                                    (i - a) : (i + b), (j - a) : (j + b)
                                ]
                                endpoints = connection(neighbors, i, j, endpoints, n=a)
                            a = a + 1
                            b = b + 1

        ridge_extra = ridge_extra + endpoints
        Result_binary = np.where(
            ridge_extra[l_c:-l_c, l_c:-l_c] > 0, 1, np.nan
        )  # Binary
        Result = np.where(
            Result_binary == 1, Result_conv, np.nan
        )  # Fill with values convolution

        end = time.time()
        print(end - start)

        # Thinning lines for further processing in qgis
        Result_binary = skeletonize(Result_binary)
        Result_binary = np.where(Result_binary == 1, 1, np.nan)

        # plt.figure(figsize=(8,8))
        # plt.imshow(Result_binary)

        # """SAVE RESULTS"""
        tif = z
        tif[0, :, :] = Result_binary
        tif.rio.to_raster(result_path / str("A2_Results_binary.tif"))

        tif = z
        tif[0, :, :] = Result
        tif.rio.to_raster(result_path / str("A2_Results.tif"))

        tif = z
        tif[0, :, :] = Rho_final
        tif.rio.to_raster(result_path / str("A2_rho.tif"))

        tif = z
        tif[0, :, :] = Sigma_final
        tif.rio.to_raster(result_path / str("A2_sigma.tif"))

        tif = z
        tif[0, :, :] = Result_conv
        tif.rio.to_raster(result_path / str("A2_v2_results_convolution.tif"))

    #%% ----------------------------------------------------------------------
    def algorithm_3(
        self,
        raster_in=None,
        result_path=None,
        N_theta=None,
        variable_sigma=None,
        threshold=None,
    ):
        """
        Algorithm 1 of Thirza van Noppen's thesis to detect high line elements in a DTM.

        Parameters
        ----------
        raster_in : str
            AHN raster file name.
        result_path : WindowsPath, optional
            Path to write the output. Default is the root path.
        N_theta : int, optional
            Number of orientations. Default is 6.
        variable_sigma : array, optional
            Kernel size. The default is array([0.2, 0.4, 0.6, 0.8, 1. ])
        threshold : int, optional
            Threshold value. The default is 15

        Returns
        -------
        None. Four TIF-files are written containing the output.

        """
        if raster_in is None:
            raster_in = self.root / "AHN" / str("AHN_TEST.TIF")
        if result_path is None:
            print("No result folder provided, writing it to root.")
            result_path = self.root

        start = time.time()

        # Parameters voor kernels
        if N_theta is None:
            N_theta = 6  # Number of orientations
        variable_theta = np.arange(0, np.pi, np.pi / N_theta)
        if variable_sigma is None:
            variable_sigma = np.arange(0.2, 1.1, 0.2)  # Size kernel
        if threshold is None:
            threshold = 15  # Threshold value

        #%% PART I: PRE-PROCESSING DATA
        # Import DTM
        logf = open(result_path / "logfile_algorithm3.txt", "w")
        logf.write("Reading input raster...\n")
        logf.close()

        z = rio.open_rasterio(raster_in)

        # Select data AHN
        tif = z[0, :, :]
        sample = np.where(tif > 1e5, np.nan, tif.values)

        # IDW > Fill nan values
        s_mask = np.where(np.isnan(sample), 0, 1)
        sample = fillnodata(sample, mask=s_mask)

        # PART II: KERNEL CONVOLUTION

        def kernel_conv(DEM, s=1, t=0, r=1, lowest=-5, highest=5, step=0.1, gamma=0.75):
            x = np.arange(highest, lowest, -step)
            y = np.arange(lowest, highest, step)
            x, y = np.meshgrid(x, y)

            # Create kernel based on sigma, theta & rho
            M_theta = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            rho_sqrt = np.array([[r ** 2, 0], [0, r ** -2]])
            z_second = np.zeros((len(x), len(y)))
            for i in range(len(x)):
                for j in range(len(y)):
                    xy = np.array([x[i, j], y[i, j]])
                    x_y = np.array([[x[i, j]], [y[i, j]]])
                    phi = xy @ M_theta.transpose() @ rho_sqrt @ M_theta @ x_y
                    G_hat = (1 / (2 * np.pi * s ** 2)) * np.exp(-phi / (2 * s ** 2))
                    SAG = (
                        (((x[i, j] * np.cos(t)) + (y[i, j] * np.sin(t))) ** 2)
                        / (r ** -4 * s ** 4)
                        - (r ** 2 / s ** 2)
                    ) * G_hat
                    z_second[i, j] = SAG

            # Convolution
            DEM_conv = -((s ** 2) ** gamma) * convolve(DEM, z_second)
            return DEM_conv

        Direction = np.zeros(np.shape(sample))
        Sigma_final = np.zeros(np.shape(sample)) + variable_sigma[0]
        logf = open(result_path / "logfile_algorithm3.txt", "a")
        logf.write("Starting loops...\n")
        logf.close()
        for j in tqdm(range(len(variable_sigma)), total=len(variable_sigma)):
            for k in range(len(variable_theta)):
                logf = open(result_path / "logfile_algorithm3.txt", "a")
                logf.write(
                    str(j)
                    + "/"
                    + str(len(variable_sigma))
                    + "; "
                    + str(k)
                    + "/"
                    + str(len(variable_theta))
                    + "\n"
                )
                logf.write("Elapsed time:" + str(time.time() - start) + "\n")
                logf.close()
                DTM_conv = kernel_conv(
                    DEM=sample, s=variable_sigma[j], t=variable_theta[k]
                )
                if k == 0 and j == 0:
                    Result_conv = DTM_conv  # Wordt uiteindelijke resultaat in opgeslagen van convolutie
                else:
                    Sigma_final = np.where(
                        DTM_conv > Result_conv, variable_sigma[j], Sigma_final
                    )
                    Direction = np.where(
                        DTM_conv > Result_conv, variable_theta[k], Direction
                    )
                    Result_conv = np.where(
                        DTM_conv > Result_conv, DTM_conv, Result_conv
                    )

        # PART III: THRESHOLDING
        Result_binary = np.where((Result_conv > threshold), 1, np.nan)

        # PART IV: SKELETONIZE
        Result_binary = skeletonize(Result_binary)
        Result = np.where(Result_binary == 1, Result_conv, np.nan)

        # Outside to np.nan
        Result_binary[0, :] = np.nan
        Result_binary[-1, :] = np.nan
        Result_binary[:, 0] = np.nan
        Result_binary[:, -1] = np.nan

        end = time.time()
        print(end - start)

        # """SAVE RESULTS"""
        # data_path = os.chdir('../Results')
        logf = open(result_path / "logfile_algorithm3.txt", "a")
        logf.write("Writing results to tif...\n")
        logf.close()
        tif = z
        tif[0, :, :] = Result_binary
        tif.rio.to_raster(result_path / str("A3_v2_Results_binary.tif"))

        tif = z
        tif[0, :, :] = Result
        tif.rio.to_raster(result_path / str("A3_v2_Results.tif"))

        tif = z
        tif[0, :, :] = Sigma_final
        tif.rio.to_raster(result_path / str("A3_v2_sigma.tif"))

        tif = z
        tif[0, :, :] = Result_conv
        tif.rio.to_raster(result_path / str("A3_v2_results_convolution.tif"))
        logf = open(result_path / "logfile_algorithm3.txt", "a")
        logf.write("Done in " + str(end - start) + "\n")
        logf.close()


if __name__ == "__main__":
    lines = DetectLineElements(path=r"D:\4632.10\data")
