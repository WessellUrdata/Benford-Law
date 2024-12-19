from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import scipy.interpolate
from tqdm import tqdm


def azimuthalAverage(magnitude_spectrum: np.ndarray) -> np.ndarray:
    # Calculate the indices from the image
    y, x = np.indices(magnitude_spectrum.shape)

    center_y, center_x = ((i - 1) / 2 for i in magnitude_spectrum.shape)

    r = np.hypot(y - center_y, x - center_x)

    # Get the integer part of the radii (bin size = 1)
    r_int = r.astype(int)

    # Calculate the mean for each radius bin
    tbin = np.bincount(r_int.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r_int.ravel())

    radial_prof = tbin / nr

    return radial_prof


class FeatureExtraction:
    def __init__(self, features: int = 100):
        self.features = features

    def fft(
        self,
        filename: str,
        crop: bool = True,
        freq_shift: bool = True,
    ) -> list[float]:
        img = cv2.imread(filename, 0)
        height, width = img.shape[:2]

        if crop:
            # we crop the center
            height = height // 3
            width = width // 3
            img = img[height:-height, width:-width]

        # do FFT
        frequencies = np.fft.fft2(img)
        # shift zero frequency component to center
        if freq_shift:
            frequencies = np.fft.fftshift(frequencies)

        # calculate magnitude spectrum
        magnitude_spectrum = np.abs(frequencies)

        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = azimuthalAverage(magnitude_spectrum)

        points = np.linspace(
            0, self.features, num=psd1D.size
        )  # coordinates of points in psd1D
        xi = np.linspace(
            0, self.features, num=self.features
        )  # coordinates for interpolation

        interpolated = scipy.interpolate.griddata(points, psd1D, xi, method="cubic")

        return interpolated

    def multithread_fft(self, filenames: list[str], **kwargs) -> list[list[float]]:
        with Pool(processes=cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(partial(self.fft, **kwargs), filenames),
                    total=len(filenames),
                    desc="Performing FFT",
                )
            )
        return results
