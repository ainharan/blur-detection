import cv2
import numpy as np
import os
import sys

def importData(path):
    """
    Imports data given a path
    :type path: string
    :params path: data path
    """
    return [f for f in os.listdir(path) if os.path.isfile(path+f)]

def variance_of_laplacian(image):
    '''
    Returns the variance of the Laplacian of an image
    :type image: cv2.image
    :params image: cv2 image file
    '''
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_min_variance(imgs, gpath):
    '''
    Returns min Laplacian variance of good images
    :type imgs: list
    :params imgs: list of image files
    :type gpath: string
    :params gpath: directory path of good images
    '''
    min_var = sys.maxint
    for i in imgs:
        img = cv2.imread(gpath + i, cv2.IMREAD_GRAYSCALE) # read image as grayscale
        var = variance_of_laplacian(img)
        if var < min_var:
            min_var = var

    return min_var

def main():
    # data paths 
    blurry_path = './data/blurry-data/'
    flare_path = './data/flare-data/'
    good_path = './data/good-data/'

    threshold = get_min_variance(importData(good_path), good_path)
    img_path = os.path.dirname(sys.argv[1])+ '/'
    img_name = os.path.basename(sys.argv[1])
    img = cv2.imread(img_path + img_name, cv2.IMREAD_GRAYSCALE)
    if variance_of_laplacian(img) < threshold:
        print "Blurry"
    else:
        print "Not blurry"

if __name__ == "__main__":
    main()
