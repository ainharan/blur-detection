import cv2
import numpy as np

class Hog(object):

    def __init__(self):
        self.bin_n = 16 # number of bins

    # from opencv documentation
    def hog(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)

        # quantizing binvalues in (0...16)
        bins = np.int32(self.bin_n*ang/(2*np.pi))

        # Divide to 4 sub-squares
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), self.bin_n) for b, m in zip(bin_cells, mag_cells)]
        # return histogram as 64 bit vector
        return np.hstack(hists)

