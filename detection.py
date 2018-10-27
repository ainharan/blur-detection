import cv2
import numpy as np
import os

def importData():
    """
    ##############
    """
    # data path 
    good_path = './data/good-data/'
    blurry_path = './data/blurry-data/'
    flare_path = './data/flare-data/'

    good = [f for f in os.listdir(good_path) if os.path.isfile(good_path+f)]
    flare = [f for f in os.listdir(flare_path) if os.path.isfile(flare_path+f)]
    blurry = [f for f in os.listdir(good_path) if os.path.isfile(blurry_path+f)]
    good1 = cv2.imread(good_path + good[0])
    cv2.imshow('first good img', good1)
    cv2.waitKey()

def variance_of_laplacian(image):
    '''
    compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian

    :type image: cv2.image ########################
    :params image: ##############
    '''
    return cv2.Laplacian(image, cv2.CV_64F).var()

def main():
    importData()


if __name__ == "__main__":
    main()
