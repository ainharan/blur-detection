import numpy as np
import cv2

from preprocessing import Preprocessing
from hog import Hog


def get_sets():
    '''
    Returns np array of labelled data
    '''
    labels = []
    training = []
    pp = Preprocessing()

    # get list of data files
    flare_list = pp.get_training_data(pp.flare_path)
    blurry_list = pp.get_training_data(pp.blurry_path)
    good_list = pp.get_training_data(pp.good_path)

    # label flare files as 1 and add to training set
    append_sets(flare_list, pp.flare_path, training, labels, 1)
    # label blurry files as 2 and add to training set
    append_sets(blurry_list, pp.blurry_path, training, labels, 2)
    # label good files as 3 and add to training set
    append_sets(good_list, pp.good_path, training, labels, 3)

    return np.float32(labels), np.float32(training)

def append_sets(dataset, path, training, labels, l):
    '''
    appends labels to labels list for an associated image
    appends histograms to training for an associated image
    '''
    h = Hog()
    for file in dataset:
        img = cv2.imread(path + file)
        # appends histogram of oriented gradients
        # to training list
        r_img=cv2.resize(img,(400,300))
        hist = h.hog(r_img)
        training.append(hist)
        # appends appropriate label to labels list
        labels.append(l)


def main():
     labels, training = get_sets()
     #print labels
     #print training

if __name__ == "__main__":
    main()
